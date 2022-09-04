def generate_lags(series_ts, k):

    import numpy as np, pandas as pd
    xs = []
    inde = series_ts.index
    for i in range(k):
        name = '{} lags {}'.format(series_ts.name, str(i+1))
        y_past = [np.nan]*(i+1) + list(series_ts[:-(i+1)])
        y_past = pd.Series(y_past, index=inde, name=name)
        xs.append(y_past)
    df_t = pd.concat([series_ts.copy(), pd.DataFrame(xs).T],axis=1)
    y = df_t.dropna().iloc[:,0]
    x = df_t.dropna().iloc[:,1:]
    return df_t, x, y

def ADF_Test(series_ts):
    '''
    Statistical Stationary Test (ADF test).
    p-value >  0.05: non-stationary, return False
    p-value <= o.o5: stationary, return True
    '''
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series_ts.values)
    print('ADF Statistic: %f' % result[0])
    if result[1] > 0.05:
        return False
    else:
        return False

class MA:
    '''
    Moving Average Model with q period lags.
    '''
    def __init__(self, q=None):
        self.lags = q
        self.name = "Moving Average with {0} lags.".format(self.lags)
    
    def set_params(self, q):
        self.lags = q
        self.name = "Moving Average with {0} lags.".format(self.lags)
    
    def fit(self, series_ts):
        self.data_train = series_ts
    
    def predict(self, series_ts):
        pred = series_ts.rolling(self.lags).mean()
        pred.name = "MA({0}) Prediction".format(self.lags)
        return pred

    def score(self, series_ts, scoring = 'mse'):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import pandas as pd
        y_true = series_ts.copy()
        y_hat = self.predict(series_ts) 
        df_temp = pd.concat([y_true, y_hat], axis=1).dropna()
        y_true = df_temp.iloc[:,0]
        y_hat = df_temp.iloc[:,1]
        if scoring == 'mse':
            return mean_squared_error(y_true, y_hat)
        if scoring == 'mae':
            return mean_absolute_error(y_true, y_hat)
        if scoring == 'r2':
            return r2_score(y_true, y_hat)
    
    def plot(self, series_ts, scoring = None):
        import matplotlib.pyplot as plt
        y_hat = self.predict(series_ts)
        y_true = series_ts.copy()
        if scoring:
            scor = self.score(series_ts, scoring=scoring)
            title = "{0}\n{1}: {2}".format(self.name, scoring, str(scor))
        else:
            title = self.name
        plt.plot(y_true)
        plt.plot(y_hat)
        plt.title(title)
        plt.show()
        
class AR:
    '''
    Auto Regression Model with p period lags.
    '''
    def __init__(self, p=None):
        self.lags = p
        self.name = "Auto Regression with {0} lags.".format(self.lags)
        self.model = None
        
    def set_params(self, p):
        self.lags = p
        self.name = "Auto Regression with {0} lags.".format(self.lags)
        
    def fit(self, series_ts):
        from sklearn.linear_model import LinearRegression
        df_t, X, Y = generate_lags(series_ts, self.lags)
        LR = LinearRegression()
        LR.fit(X, Y)
        self.model = LR
        self.intercept_= LR.intercept_
        self.coef = LR.coef_
  
    def predict(self, series_ts):
        import pandas as pd, numpy as np
        X = generate_lags(series_ts, self.lags)[1]
        y_hat = np.append(np.full(self.lags, np.nan), self.model.predict(X))
        return pd.Series(y_hat, index=series_ts.index, name="AR({0}) Prediction".format(self.lags))

    def score(self, series_ts, scoring = 'mse'):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        y_hat = self.predict(series_ts).dropna()
        y_true = generate_lags(series_ts, self.lags)[2]
        n = y_true.size
        k = self.lags+1
        if scoring == 'mse':
            return mean_squared_error(y_true, y_hat)
        if scoring == 'mae':
            return mean_absolute_error(y_true, y_hat)
        if scoring == 'r2':
            return r2_score(y_true, y_hat)
        if scoring == 'adj_r2':
            return 1-(1-r2_score(y_true, y_hat))*((n-1)/(n-k))
    
    def plot(self, series_ts, scoring=None):
        '''
        If scoring = None, it does not show score. Accetible input: r2, mse, mae
        '''
        if self.model:
            import matplotlib.pyplot as plt
            y_hat = self.predict(series_ts)
            y_true = series_ts
            if scoring:
                scor = self.score(series_ts, scoring=scoring)
                title = "{0}\n{1}: {2}".format(self.name, scoring, str(scor))
            else:
                title = self.name
            plt.plot(y_true)
            plt.plot(y_hat)
            plt.title(title)
            plt.show()
        else:
            print("Model is not trained. Use '.fit' before plot")

class ARMA:
    '''
    ARMA Model with p and q periods lags.
    '''
    def __init__(self, p=None, q=None):
        self.order = [p, 0, q]
        self.name = 'ARMA with p={0}, q={1}.'.format(self.order[0], self.order[2])
        self.model = None
        
    def set_params(self, p, q):
        self.order = [p, 0, q]
        self.name = 'ARMA with p={0}, q={1}.'.format(self.order[0], self.order[2])
            
    def fit(self, series_ts):
        from statsmodels.tsa.arima.model import ARIMA
        self.model = ARIMA(series_ts, order=self.order).fit()
    
    def predict(self, series_ts):
        pred = self.model.apply(series_ts).fittedvalues
        pred.name = "ARMA({0}, {1}) Prediction".format(self.order[0], self.order[2])
        return pred
    
    def forecast(self, period):
        fore = self.model.forecast(period)
        return fore
    
    def score(self, series_ts, scoring = 'mse'):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        y_true = series_ts[1:]
        y_hat = self.predict(series_ts)[1:]
        if scoring == 'mse':
            return mean_squared_error(y_true, y_hat)
        if scoring == 'mae':
            return mean_absolute_error(y_true, y_hat)
        if scoring == 'r2':
            return r2_score(y_true, y_hat)
    
    def plot(self, series_ts, scoring=None):
        import matplotlib.pyplot as plt
        y_hat = self.predict(series_ts)
        y_true = series_ts
        if scoring:
            scor = self.score(series_ts, scoring=scoring)
            title = "{0}\n{1}: {2}".format(self.name, scoring, str(scor))
        else:
            title = self.name
        plt.plot(y_true)
        plt.plot(y_hat)
        plt.title(title)
        plt.show()

class ETS:
    '''
    Expontential Smoothing method.
    '''
    def __init__(self):
        self.name = "Expontential Smoothing."
        self.par = None
    
    def set_params(self):
        pass
    
    def fit(self, series_ts):
        import pandas as pd
        from scipy.optimize import minimize
        Y = series_ts
        alp_0 = 0.01
        def best_alpha(alp):
            nonlocal Y
            y_predicted = [Y[0]]
            for i in range(1,Y.size):
                y_h = y_predicted[i-1] + alp*(Y[i-1]-y_predicted[i-1])
                y_predicted.append(y_h)
            y_predicted = pd.Series(y_predicted, index=Y.index)
            rss = (Y[1:]-y_predicted[1:])**2
            mse = rss.mean()
            #print(y_predicted)
            return mse
        bou = [(0.0, 1.0)]
        opt_resu = minimize(best_alpha, alp_0, bounds=bou)
        self.par = opt_resu.x[0]
        
    def predict(self, series_ts):
        import pandas as pd
        y_true = series_ts
        y_predicted = [y_true[0]]
        for i in range(1,y_true.size):
            y_h = y_predicted[i-1] + self.par * (y_true[i-1]-y_predicted[i-1])
            y_predicted.append(y_h)
        y_predicted = pd.Series(y_predicted, index=y_true.index, name="ETS Prediction")
        return y_predicted
    
    def score(self, series_ts, scoring = 'mse'):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        y_hat = self.predict(series_ts)[1:]
        y_true = series_ts[1:]
        if scoring == 'mse':
            return mean_squared_error(y_true, y_hat)
        if scoring == 'mae':
            return mean_absolute_error(y_true, y_hat)
        if scoring == 'r2':
            return r2_score(y_true, y_hat)
    
    def plot(self, series_ts, scoring=None):
        '''
        If scoring = None, it does not show score. Accetible input: r2, mse, mae
        '''
        if self.par:
            import matplotlib.pyplot as plt
            y_hat = self.predict(series_ts)
            y_true = series_ts
            if scoring:
                scor = self.score(series_ts, scoring=scoring)
                title = "{0}\n{1}: {2}".format(self.name, scoring, str(scor))
            else:
                title = self.name
            plt.plot(y_true)
            plt.plot(y_hat)
            plt.title(title)
            plt.show()
        else:
            print("Model is not trained. Use .fit before plot")
    
class DETS:
    '''
    Double Expontential Smoothing method.
    '''
    def __init__(self, *profile):
        self.name = "Double expontential smoothing" 
        self.par = None
        
    def set_params(self, *profile):
        pass
    
    def fit(self, series_ts):
        import numpy as np, pandas as pd
        from scipy.optimize import minimize
        Y = series_ts
        alp_0 = 0.5
        beta_0 = 0.5
        par_0 = [alp_0, beta_0]
        def best_alpha_beta(ab):
            nonlocal Y
            a = ab[0]
            b = ab[1]
            e = [Y[0]]
            t = [0]
            y_h = [np.nan]
            for i in range(1,Y.size):
                j = i-1
                y_i_hat = e[j] + t[j]
                y_h.append(y_i_hat)
                e_i = a*Y[i] + (1-a)*(e[j]+t[j])
                e.append(e_i)
                t_i = b*(e[i]-e[j]) + (1-b)*t[j]
                t.append(t_i)
            y_h = pd.Series(y_h, index=Y.index)
            rss = (Y[1:]-y_h[1:])**2
            mse = rss.mean()
            return mse

        bou = [(0.0, 1.0), (0.0, 1.0)]
        opt_resu = minimize(best_alpha_beta, par_0, bounds=bou)
        self.par = opt_resu.x  
        
    def predict(self, series_ts):
        import pandas as pd, numpy as np
        Y = series_ts
        a = self.par[0]
        b = self.par[1]
        e = [Y[0]]
        t = [0]
        pred = [np.nan]
        for i in range(1,Y.size):
            j = i-1
            y_i_hat = e[j] + t[j]
            pred.append(y_i_hat)
            e_i = a*Y[i] + (1-a)*(e[j]+t[j])
            e.append(e_i)
            t_i = b*(e[i]-e[j]) + (1-b)*t[j]
            t.append(t_i)
        pred = pd.Series(pred, index=Y.index, name='DETS Prediction')
        return pred
    
    def score(self, series_ts, scoring = 'mse'):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        y_hat = self.predict(series_ts)[1:]
        y_true = series_ts[1:]
        if scoring == 'mse':
            return mean_squared_error(y_true, y_hat)
        if scoring == 'mae':
            return mean_absolute_error(y_true, y_hat)
        if scoring == 'r2':
            return r2_score(y_true, y_hat)

    
    def plot(self, series_ts, scoring=None):
        '''
        If scoring = None, it does not show score. Accetible input: r2, mse, mae
        '''
        if self.par.size:
            import matplotlib.pyplot as plt
            y_hat = self.predict(series_ts)
            y_true = series_ts
            if scoring:
                scor = self.score(series_ts, scoring=scoring)
                title = "{0}\n{1}: {2}".format(self.name, scoring, str(scor))
            else:
                title = self.name
            plt.plot(y_true)
            plt.plot(y_hat)
            plt.title(title)
            plt.show()
        else:
            print("Model is not trained. Use .fit before plot")        
    
class DMA:
    '''
    Double Moving Average method.
    '''
    def __init__(self, p=None):
        self.lags = p
        self.name = "Double Moving Average with {0} lags.".format(self.lags) 
        
    def set_params(self, p):
        self.lags = p
        self.name = "Double Moving Average with {0} lags.".format(self.lags) 
    
    def fit(self, series_ts):
        self.data_train = series_ts
        pass
        
    def predict(self, series_ts):
        import pandas as pd, numpy as np
        k = self.lags
        j = 2*k-1
        df_t,x,y = generate_lags(series_ts, j)
        m = x.iloc[:,0:k].mean(axis=1)
        d = pd.Series(0.0, index=x.index)
        for i in range(k):
            start = 0+i
            end = k+i
            d += x.iloc[:,start:end].mean(axis=1)
        d = d/k
        x['M_t'] = m
        x['D_t'] = d
        x['E_t'] = 2*x['M_t'] - x['D_t']
        x['T_t'] = 2*(x['M_t'] - x['D_t'])/(k-1)
        x['Y_hat'] = x['E_t'] + x['T_t']
        pred = pd.concat([pd.Series([np.nan]).repeat(j), x['Y_hat']], ignore_index=True)
        pred.index = series_ts.index
        pred.name="DMA({0}) Prediction".format(self.lags)
        return pred
    
    def score(self, series_ts, scoring = 'mse'):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        j = 2*self.lags-1
        y_true = series_ts[j+1:]
        y_hat = self.predict(series_ts)[j+1:]
        if scoring == 'mse':
            return mean_squared_error(y_true, y_hat)
        if scoring == 'mae':
            return mean_absolute_error(y_true, y_hat)
        if scoring == 'r2':
            return r2_score(y_true, y_hat)

    
    def plot(self, series_ts, scoring=None):
        '''
        If scoring = None, it does not show score. Accetible input: r2, mse, mae
        '''
        import matplotlib.pyplot as plt
        y_hat = self.predict(series_ts)
        y_true = series_ts
        if scoring:
            scor = self.score(series_ts, scoring=scoring)
            title = "{0}\n{1}: {2}".format(self.name, scoring, str(scor))
        else:
            title = self.name
        plt.plot(y_true)
        plt.plot(y_hat)
        plt.title(title)
        plt.show()

class ARIMA:
    '''
    ARIMA Model with p, d, and q periods lags.
    '''
    def __init__(self, p=None, d=None, q=None):
        self.order = [p,d,q]
        self.name = 'ARIMA with p={0}, d={1}, q={2}'.format(self.order[0], self.order[1], self.order[2])
        self.model = None
        
    def set_params(self, p=None, d=None, q=None):
        self.order = [p,d,q]
        self.name = 'ARIMA with p={0}, d={1}, q={2}'.format(self.order[0], self.order[1], self.order[2])
    
    def fit(self, series_ts):
        from statsmodels.tsa.arima.model import ARIMA
        self.model = ARIMA(series_ts, order=self.order).fit()
    
    def predict(self, series_ts):
        pred = self.model.apply(series_ts).fittedvalues
        pred.name = "ARIMA({0}, {1}, {2}) Prediction".format(self.order[0], self.order[1], self.order[2])
        return pred
    
    def score(self, series_ts, scoring = 'mse'):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        y_true = series_ts[1:]
        y_hat = self.predict(series_ts)[1:]
        if scoring == 'mse':
            return mean_squared_error(y_true, y_hat)
        if scoring == 'mae':
            return mean_absolute_error(y_true, y_hat)
        if scoring == 'r2':
            return r2_score(y_true, y_hat)
    
    
    def plot(self, series_ts, scoring=None):
        import matplotlib.pyplot as plt
        y_hat = self.predict(series_ts)
        y_true = series_ts
        if scoring:
            scor = self.score(series_ts, scoring=scoring)
            title = "{0}\n{1}: {2}".format(self.name, scoring, str(scor))
        else:
            title = self.name
        plt.plot(y_true)
        plt.plot(y_hat)
        plt.title(title)
        plt.show()
        
def gen_ts_cv(series_ts, n_split=5):
    '''
    Generating the test-validation set combinations for cross-validation.
    '''
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_split)
    train_test_comb = []
    for train_index, test_index in tscv.split(series_ts):
        comb = [series_ts[train_index], series_ts[test_index]]
        train_test_comb.append(comb)
    return train_test_comb

class GridSearchCV_TS:
    '''
    Hyperparameters Tuning and Cross-validation(default cv=5) for models, supporting multiprocessing(default n_jobs=1).
    '''
    def __init__(self, model, param_dic, cv, n_jobs):
        import os
        self.model = model
        self.n_split = cv
        self.process = n_jobs
        self.para_dic = param_dic
        if n_jobs == -1:
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = n_jobs
    
    def fit(self, series_ts):
        import multiprocessing as mp, copy
        from sklearn.model_selection import ParameterGrid
        global CV_score
        parm_lis = list(ParameterGrid(self.para_dic))
        par_try = []
        for par_dic in parm_lis:
            m = copy.deepcopy(self.model)
            n_spl = copy.deepcopy(self.n_split)
            l = [m, n_spl, series_ts] + list(par_dic.values())
            par_try.append(l)
        self.par_try = par_try
        def CV_score(model, cv, series_ts, *p_lis):
            from statistics import mean
            print(p_lis)
            lis_score = []
            model.set_params(*p_lis)
            lis_cv = gen_ts_cv(series_ts, cv)
            for train, test in lis_cv:
                model.fit(train)
                s = model.score(test)
                lis_score.append(s)
            score_av = mean(lis_score)
            print(score_av, p_lis)
            return [score_av, p_lis]
        if __name__ == "__main__":
            pool = mp.Pool(self.n_jobs)
            result = pool.starmap(CV_score, par_try)
            pool.close()
            pool.join()
            best = min(result, key=lambda x: x[0])
            print(best)
            self.result = result
            self.best = result[0]
            self.best_params_ = best[1]
            self.best_score_ = best[0]
            self.best_estimator = copy.copy(self.model)
            self.best_estimator.set_params(*best[1])
            self.best_estimator.fit(series_ts)


