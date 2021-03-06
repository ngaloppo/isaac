# Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
# 
# This file is part of ISAAC.
# 
# ISAAC is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301  USA


import random, argparse, json, os
from math import log, isinf
from itertools import chain, product
from numpy import argsort, argmax, where, delete, bincount
from operator import mul
import isaac as sc
from tools import profile_execution_failure
from isaac.external.sklearn.forest import RandomForestRegressor
import optimize, tools, model
from json import encoder
import json, csv
import numpy as np


encoder.FLOAT_REPR = lambda o: format(o, '.2f')
encoder.separators = (',',':')

def unique(L):
    seen = set()
    seen_add = seen.add
    return [ x for x in L if not (x in seen or seen_add(x))]

def pow2range(a, b):
    return [2**x for x in range(a, b)]


class Tuner:

    def __init__(self, logger, device, operation, json_path, progress_bar):
        self.logger = logger
        self.device = device
        self.operation = operation
        self.json_path = json_path
        self.progress_bar = progress_bar
        
  
    def run(self, level = 'intermediate'): 
        
        assert level in ['simple', 'intermediate', 'full']
        
        device = self.device
        operation = self.operation
        context = sc.driver.context(device)
        
        if self.logger:
            self.logger.info("----------------")
            self.logger.info(operation.__name__.replace('_','-').upper())
            self.logger.info("----------------")

        #BLAS1 training sizes
        if operation in [sc.templates.elementwise_1d, sc.templates.reduce_1d]:
            if level=='simple':
                sizes = [(10000000,)]
            elif level=='intermediate':
                sizes = [(x,) for x in tools.expspace(1e3, 1e8, 10)]
            else:
                sizes = [(x,) for x in tools.expspace(1e3, 1e8, 100)] 
        
        #BLAS2 training sizes
        if operation in [sc.templates.elementwise_2d, sc.templates.reduce_2d_rows, sc.templates.reduce_2d_cols]:
            if level=='simple':
                sizes = [(1536, 1536)]
            elif level=='intermediate':
                sizes = [(896,896),
                         (1536,1536),
                         (256, 256),
                         (1024,256), 
                         (4096,256),
                         (16384,256), 
                         (256,1024), 
                         (256,4096),
                         (256,16384),
                         (3025,96)]
            else:
                sizes = product(pow2range(4,17), pow2range(4,17))
        
        #BLAS3 training sizes
        if operation in [sc.templates.matrix_product_nn, sc.templates.matrix_product_nt, sc.templates.matrix_product_tn, sc.templates.matrix_product_tt]:
            if level=='simple':
                sizes = [(2560,2560,2560)]
            elif level=='intermediate':
                sizes = [#Square
                         (896,896,896),
			 (1536,1536,1536),
			 (2176, 2176,2176),
                         #Rank-32 updates
                         (896,896,32),
                         (1536,1536,32),
                         (2176,2176,32),
                         #Covariance
			 (32,32,16000),
			 (64,64,64000),
                         (256,256,32000),
                         #Convolutions
                         (3025,64,363),
                         (729,192,1200),
                         (169,384,1728),
                         (169,256,3456),
                         (169,128,2304),
                         (169,2304,256),
                         (169,3456,256),
                         (169,1728,384),
                         (729,1600,192),
                         (3025,363,64), 
                         (2304,256,169),
                         (3456,256,169),
                         (1728,384,169),
                         (1600,192,729),
                         (363,64,3025)]
            elif level=='full':
			    sizes = product(pow2range(5, 12), pow2range(5, 12), pow2range(5, 17))

        #Remove duplicates and or too small/big tuples
        sizes = [x for x in sizes if 1e-4 <= tools.memory_footprint(operation, x) <= 2e-1]
        
        #Training data
        performance = tools.metric_of(operation)
        profiles, X, Y = [], [], []
        
        #Restore previous run
        savepath = os.path.join('save', operation.__name__)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        try:
            with open(os.path.join(savepath, 'X.csv')) as f:
                X = [tuple(map(int, row)) for row in csv.reader(f, delimiter=',')]
                
            with open(os.path.join(savepath, 'Y.csv')) as f:
                Y = [map(float, row) for row in csv.reader(f, delimiter=',')]
            
            with open(os.path.join(savepath, 'profiles.csv')) as f:
                def mmap(x):
                    if x=='FETCH_FROM_LOCAL':
                        return sc.templates.fetch_type.FETCH_FROM_LOCAL
                    if x=='FETCH_FROM_GLOBAL_CONTIGUOUS':
                        return sc.templates.fetch_type.FETCH_FROM_GLOBAL_CONTIGUOUS
                    if x=='FETCH_FROM_GLOBAL_STRIDED':
                        return sc.templates.fetch_type.FETCH_FROM_GLOBAL_STRIDED
                    return int(x)
                profiles = [map(mmap,row) for v in row for row in csv.reader(f, delimiter=',')]
        except:
            pass
        
        ##### Exploration #####
        for idx, x in enumerate(sizes):
            if idx>0:
                self.progress_bar.set_finished()

            self.progress_bar.set_prefix(', '.join(map(str, x)))
            #Skip if saved
            if x in X:
                row = Y[X.index(x)]
                self.progress_bar.update(1, 1, profiles[argmax(row)], max(row))
                continue
            
            #Check if the current best prediction is not a local optimum
            idx = len(X)
            nparams = len(profiles)
            tree, operands = tools.tree_of(operation, x, context)
            if idx==0:
                retune = True
                predicted = None
            else:
                if nparams==1:
                    predicted = profiles[0]
                else:
                    clf = RandomForestRegressor(min(10, idx+1), max_depth=min(10, idx+1)).fit(X, Y)
                    #clf, nrmse = model.train(X, Y, profiles)
                    predperf = clf.predict(x)[0]
                    best = (-predperf).argsort()[:5]
                    perf = []
                    for b in best:
                        try:
                            perf += [performance(x, tools.benchmark(operation, profiles[b], tree))]
                        except profile_execution_failure:
                            pass
                    predicted = profiles[best[argmax(perf)]]
                retune = not optimize.is_local_optimum(predicted, operation, x, context)
                
            #Retune if necessary
            if retune:
                optimizer = optimize.GeneticOptimizer(self.logger, naccept=1000, niter=1000, cxpb=.4, mutpb=.4, popsize=20, progress_bar = self.progress_bar)
                new = optimizer.run(operation, x, context, prior=predicted)[0]
                if new not in profiles:
                    profiles.append(new)
                    if idx > 0:
                        for xx,yy in zip(X, Y):
                            _tree, _operands = tools.tree_of(operation, xx, context)
                            try:
                                time = tools.benchmark(operation, new, _tree)
                                perf = performance(xx, time)
                            except profile_execution_failure:
                                perf = 0
                            yy.append(0 if isinf(perf) else perf)
                            
                
            ##### Training #####
            y = []
            fastest = max(predperf) if nparams > 1 else None
            for ip, p in enumerate(profiles):
                try:
                    perf = 0 if fastest and ip < nparams and predperf[ip]/fastest < .1 else performance(x,tools.benchmark(operation, p, tree))
                except profile_execution_failure:
                    perf = 0
                y.append(0 if isinf(perf) else perf)
            X.append(x)
            Y.append(y)
            
            #Save data
            for (fname, data) in zip(['X.csv', 'Y.csv', 'profiles.csv'], [X, Y, profiles]):
                with open(os.path.join(savepath, fname), 'wb') as f:
                    csv.writer(f).writerows(data)
            
            #print performance info in case no tuning was done
            if not retune:
                row = Y[X.index(x)]
                self.progress_bar.update(1, 1, profiles[argmax(row)], max(row))
        self.progress_bar.set_finished()
        
        #Remove unused profiles
        if len(Y[0]) > 1:
            unused = np.where(np.bincount(np.argmax(Y, 1))==0)[0]
            profiles = [p for ip,p in enumerate(profiles) if ip not in unused]
            Y = np.delete(Y, np.where(np.bincount(np.argmax(Y, 1))==0), axis=1).tolist()          
        
        ##### Exportation #####
        json_path = tools.sanitize(device.name) + '.json' if not self.json_path else self.json_path
        if os.path.isfile(json_path):
            json_data = json.load(open(json_path, 'r'))
        else:
            json_data = {}
            json_data["version"] = "1.0"
        operation_name = operation.__name__
        if operation_name not in json_data:
            json_data[operation_name] = {}
        json_data[operation_name]['float32'] = {}
        D = json_data[operation_name]['float32']
        if len(profiles) > 1:
            clf, nrmse = model.train(X, Y, profiles)
            D['predictor'] = [{'children_left': e.tree_.children_left.tolist(),
                                'children_right': e.tree_.children_right.tolist(),
                                'threshold': e.tree_.threshold.astype('float64').tolist(),
                                'feature': e.tree_.feature.astype('float64').tolist(),
                                'value': e.tree_.value[:,:,0].astype('float64').tolist()} for e in clf.estimators_]
        D['profiles'] = [map(int, x) for x in profiles]
        json.dump(json_data, open(json_path,'w'))
