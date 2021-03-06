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

from isaac.external.sklearn.forest import RandomForestRegressor
import numpy as np

def gmean(a, axis=0, dtype=None):
    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the default dtype allowing array type
        if isinstance(a,np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))

def nrmse(y_ground, y):
    N = y.size
    rmsd = np.sqrt(np.sum((y_ground - y)**2)/N)
    if len(y_ground) > 1:
        return rmsd/(np.max(y_ground) - np.min(y_ground))
    else:
        return rmsd

def train(X, Y, profiles):      
    X = np.array(X)
    Y = np.array(Y)
    #Shuffle
    p = np.random.permutation(X.shape[0])
    M = X.shape[0]
    X = X[p,:]
    Y = Y[p,:]   

    #Train the.profile
    cut = int(.7*M)
    XTr, YTr = X[:cut,:], Y[:cut,:]
    XCv, YCv = X[cut:,:], Y[cut:,:]

    nrmses = {}
    for N in range(1,min(M+1,20)):
        for depth in range(1,min(M+1,20)):
            clf = RandomForestRegressor(N, max_depth=depth).fit(XTr, YTr)
            t = np.argmax(clf.predict(XCv), axis = 1)
            y = np.array([YCv[i,t[i]] for i in range(t.size)])
            ground = np.max(YCv[:,:], axis=1)
            nrmses[clf] = nrmse(ground, y)
         
    clf = min(nrmses, key=nrmses.get)
    return clf, nrmses[clf]
