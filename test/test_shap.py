def linear_func(x):
    y = 0.
    for i in range(x.shape[1]):
        y += (i+1)*x[:,i]
    return y

def linear_func(x):
    y = 0.
    for i in range(x.shape[1]):
        y += x[:,i]
    return y

y_train = linear_func(x_train).reshape(-1,1)
y_test = linear_func(x_test).reshape(-1,1)
sdss.y = linear_func(sdss.x).reshape(-1,1)

eagle.yscaler = MinMaxScaler(feature_range=(-1,1)).fit(np.vstack((y_train, y_test)))#StandardScaler())#
y_train = eagle.yscaler.transform(y_train)
y_test = eagle.yscaler.transform(y_test)
sdss.y = eagle.yscaler.transform(sdss.y)
