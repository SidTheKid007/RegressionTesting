from flask import Flask, render_template, request, session
from flask_session import Session
from flask_uploads import UploadSet, configure_uploads, ALL
from flask_socketio import SocketIO
import numpy as np
import pandas as pd
import json
from os import path, remove
from glob import glob
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import plotly
import plotly.graph_objs as go


app = Flask(__name__)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)


csvdatafiles = UploadSet('data', ALL)
app.config['UPLOADED_DATA_DEST'] = 'static/data'
configure_uploads(app, csvdatafiles)


socketio = SocketIO(app)


@app.route('/')
def homeForm():
    session.clear()
    return render_template('home-form.html')


@app.route('/', methods=['POST'])
def loadMain():
    # fixflask_uploads.UploadNotAllowed error
    try:
        maindata = 'static/data/' + csvdatafiles.save(request.files['maindata'])
    except Exception:
        maindata = 'static/data/' + str(request.form['maindata'])
    #try:
    #   future = 'static/data/' + csvdatafiles.save(request.files['future'])
    #except Exception:
    #   if request.form.get('future') is None:
    #       future = 'static/data/' + ''
    #   else:
    #       future = 'static/data/' + request.form.get('future')
    checkflag = validate(maindata)
    #futureflag = validate(future)
    if checkflag[0] == True:
        results = fullAnalysis(maindata)
        allvars = session["fulldata"].columns.values
        cleanvars = session["normdata"].columns.values
        target = cleanvars[-1]
        if path.exists(maindata):
            remove(maindata)
        #if futureflag == True:
            #futflag = futureCheck(maindata, future)
            #bestalgo = results[15]
            #if futflag == True:
                #fullPredict(future, bestalgo)
                # instead of this 
        return render_template('index.html', sumtables=results[0], howcleanedvars=results[1], allvars=allvars, cleanvars=cleanvars, target=target, rawdistplot=results[2], cleandistplot=results[3], heatmap=results[4], rfelist=results[5], linreggraph=results[6], linregtable=results[7], knngraph=results[8], knntable=results[9], extreegraph=results[10], extreetable=results[11], xgboostgraph=results[12], xgboosttable=results[13], finalsummary=results[14])
    else:
        if path.exists(maindata):
            remove(maindata)
        return render_template('error-page.html')


@app.route('/example')
def loadExample():
    maindata = 'static/example/avocadoCA.csv'
    results = fullAnalysis(maindata)
    allvars = session["fulldata"].columns.values
    cleanvars = session["normdata"].columns.values
    target = cleanvars[-1]
    return render_template('index.html', sumtables=results[0], howcleanedvars=results[1], allvars=allvars, cleanvars=cleanvars, target=target, rawdistplot=results[2], cleandistplot=results[3], heatmap=results[4], rfelist=results[5], linreggraph=results[6], linregtable=results[7], knngraph=results[8], knntable=results[9], extreegraph=results[10], extreetable=results[11], xgboostgraph=results[12], xgboosttable=results[13], finalsummary=results[14])


def clearCache():
    #cachefiles = glob('__pycache__/*')
    #for c in cachefiles:
    #    remove(c)
    datafiles = glob('static/data/*')
    for d in datafiles:
        remove(d)


def validate(maindata):
    # change to methods to accomadate diff inputs
    checkflagcsv = False
    filetype = ''
    try:
        dfcheck = pd.read_csv(maindata)
        filetype = 'csv'
        targetvar = dfcheck.columns.values[-1]
        if (len(dfcheck.columns.values) > 1):
            # try numeric conversion?
            checkflagcsv = True
            # make diff error page for this?
    except Exception:
        checkflagcsv = False
        #try:
        #    dfcheck = pd.read_excel(maindata)
        #    filetype = 'excel'
        #    targetvar = dfcheck.columns.values[-1]
        #    if (dfcheck[targetvar].dtype == np.float64 or dfcheck[targetvar].dtype == np.int64):
        #        if (len(dfcheck.columns.values) > 1):
        #            checkflagexcel = True
        #            dfcheck.to_csv(maindata[:-5] + '.csv', encoding='utf-8')
        #            remove(maindata)
                    # delete file
        #except Exception:
        #    checkflagexcel = False
    checkflag = [checkflagcsv, filetype]
    return checkflag


def futureCheck(maindata, future):
    # do more diligence here (ex. same data types instead of names then rename. check order, etc)
    checkflag = False
    main = pd.read_csv(maindata)
    fut = pd.read_csv(future) 
    maincols = main.columns.values
    futcols = fut.columns.values
    if (np.array_equal((maincols[:-1]),futcols)):
        checkflag = True
    else:
        checkflag = False
    return checkflag


def fullAnalysis(maindata):
    fulldata = readFile(maindata)
    overview = makeOverview(fulldata)
    cleandata = cleanColumns(fulldata)
    # make diff error pages with the same template and link them (tar var, not enough data)
    # outdata = outliers(cleandata) - optional <remove outliers>
    normdata = normalize(cleandata)
    howcleanedvars = revealClean()
    rawdistplot = makeDists(fulldata, fulldata.columns.values[0])
    cleandistplot = makeDists(normdata, normdata.columns.values[0])
    heatmap = corrMatrix(normdata)
    rfelist = ''
    if (len(normdata.columns.values) > 2):
        rfelist = rfeAlgo(normdata)
    # rfe clean or norm data?
    splitdata = splitTrainTest(normdata, .8)
    linreg = linearRegression(splitdata)
    linreggraph = graphModelResults(linreg, 'Training')
    linregresults = metricModelResults(linreg, normdata)
    linregtable = linregresults[1]
    knn = kNearest(splitdata)
    knngraph = graphModelResults(knn, 'Training')
    knnresults = metricModelResults(knn, normdata)
    knntable = knnresults[1]
    extree = extraTrees(splitdata)
    extreegraph = graphModelResults(extree, 'Training')
    extreeresults = metricModelResults(extree, normdata)
    extreetable = extreeresults[1]
    xgboost = xgBoost(splitdata)
    xgboostgraph = graphModelResults(xgboost, 'Training')
    xgboostresults = metricModelResults(xgboost, normdata)
    xgboosttable = xgboostresults[1]
    fullsum = fullSummary(linregresults[0], knnresults[0], extreeresults[0], xgboostresults[0])
    summary = fullsum[0]
    bestalgo = fullsum[1]
    results = [overview, howcleanedvars, rawdistplot, cleandistplot, heatmap, rfelist, linreggraph, linregtable, knngraph, knntable, extreegraph, extreetable, xgboostgraph, xgboosttable, summary, bestalgo]
    return results


def fullPredict(future, bestalgo):
    futdata = readFile(future)
    # do further checks of validity of dataset here
    cleandata = cleanFuture(futdata)
    predictions = predictData(cleandata, bestalgo)
    saveData(future, predictions, bestalgo)


def readFile(maindata):
    # clean string for common errors
    fulldata = pd.read_csv(maindata)
    # check file type and read diff types
    if (fulldata.columns.values[0] == 'Unnamed: 0'):
        fulldata = fulldata.drop(['Unnamed: 0'], axis=1)
    # add logic to check if col 1 is an index then fulldata = fulldata.set_index(fulldata.columns.values[0])?
    return fulldata


def makeOverview(fulldata):
    # Clean up and extract commons later
    # Add names for boxes
    summary =  fulldata.describe(include = 'all').T
    sumlabels = summary.columns.values
    if ('unique' in sumlabels) & ('mean' in sumlabels):
        summaries = mixSummarize(fulldata)
    elif ('unique' in sumlabels):
        summaries = discSummarize(fulldata)
    else:
        summaries = contSummarize(fulldata)
    return summaries


def contSummarize(fulldata):
    summary =  fulldata.describe(include = 'all').T
    summary['nulls'] = fulldata.isna().sum().values
    dtypes = []
    for k in fulldata.columns.values:
        dtypes.append(fulldata[k].dtype)
    summary['dType'] = dtypes
    sum1 = summary[(summary['dType'] == 'float64') | (summary['dType'] == 'int64')].dropna(axis='columns')
    sum1 = sum1.to_html(classes='sumcont2')
    summaries = [sum1]
    return summaries


def discSummarize(fulldata):
    summary =  fulldata.describe(include = 'all').T
    summary = summary.drop(['top','freq'], axis=1)
    summary['nulls'] = fulldata.isna().sum().values
    dtypes = []
    for k in fulldata.columns.values:
        dtypes.append(fulldata[k].dtype)
    summary['dType'] = dtypes
    sum2 = summary[(summary['dType'] != 'float64') & (summary['dType'] != 'int64')].dropna(axis='columns')
    sum2 = sum2.to_html(classes='sumdisc')
    summaries = [sum2]
    return summaries


def mixSummarize(fulldata):
    summary =  fulldata.describe(include = 'all').T
    summary = summary.drop(['top','freq'], axis=1)
    summary['nulls'] = fulldata.isna().sum().values
    dtypes = []
    for k in fulldata.columns.values:
        dtypes.append(fulldata[k].dtype)
    summary['dType'] = dtypes
    sum1 = summary[(summary['dType'] == 'float64') | (summary['dType'] == 'int64')].dropna(axis='columns')
    sum2 = summary[(summary['dType'] != 'float64') & (summary['dType'] != 'int64')].dropna(axis='columns')
    if (len(sum1) > len(sum2)):
        sum1 = sum1.to_html(classes='sumcont2')
        sum2 = sum2.to_html(classes='sumdisc2')
    else:
        sum1 = sum1.to_html(classes='sumcont')
        sum2 = sum2.to_html(classes='sumdisc')
    summaries = [sum1, sum2]
    return summaries


def cleanColumns(fulldata):
    # Clean target var too
    session["fulldata"] = fulldata
    cleandata = fulldata.copy()
    cleandata = cleandata.sort_index(axis = 0) 
    varnames = cleandata.columns.values
    targetvar = varnames[-1]
    targetdata = cleandata[targetvar].values
    cleandata = cleandata.drop([targetvar], axis=1)
    nacols = cleandata.isna().sum().values
    session["nullvar"] = []
    for k in range(len(nacols)):
        if (nacols[k]/len(cleandata) > .3):
            dropvar = varnames[k]
            cleandata = cleandata.drop([dropvar], axis=1)
            session["nullvar"].append(dropvar)
            # do some fillna here instead? (if # of cols dropped here is too much)
    session["datetimes"] = []
    session["smallDisc"] = []
    session["medDisc"] = []
    session["bigDisc"] = []
    for varname in cleandata.columns.values:
        if(cleandata[varname].dtype != np.float64 and cleandata[varname].dtype != np.int64):
            try:
                cleandata[varname] = pd.to_numeric(cleandata[varname])
            except Exception:
                cleandata = discClean(cleandata, varname)
    cleandata = cleandata.sort_index(axis = 1) 
    cleandata[targetvar] = targetdata
    cleandata = cleandata.dropna()
    session["tarDisc"] = []
    cleandata = cleanTarget(cleandata, targetvar)
    # if data is too big and date is empty shuffle and head it (aka replace 5000 )
    if (len(cleandata.columns.values) > 10):
        cleandata = cleandata.head(5000)
    return cleandata


def discClean(cleandata, varname):
    # do other checks to ensure data is a datetime
    try:
        cleandata[varname] =  pd.to_datetime(cleandata[varname])
        cleandata = datetimeDisc(cleandata, varname)
    except Exception:
        # instead of size, change it to nominal vs ordinal
        univals = len(cleandata[varname].unique())
        # change upeer bound of .2 based on research 
        if ((univals < 2) or (univals/len(cleandata) > .2)) and (len(cleandata.columns.values) > 2):
            cleandata = bigDisc(cleandata, varname)
        elif (univals < 21):
            cleandata = smallDisc(cleandata, varname)
        else:
            cleandata = medDisc(cleandata, varname)
    return cleandata


def datetimeDisc(cleandata, varname):
    cleandata[varname + '_year'] = cleandata[varname].dt.year
    cleandata[varname + '_month'] = cleandata[varname].dt.month
    cleandata[varname + '_day'] = cleandata[varname].dt.day
    # add others? (ex. season, day of year, day of week?)
    cleandata = cleandata.drop([varname], axis=1)
    session["datetimes"].append(varname)
    return cleandata


def bigDisc(cleandata, varname):
    cleandata = cleandata.drop([varname], axis=1)
    # do some fillna here? (if # of cols dropped here is too much)
    session["bigDisc"].append(varname)
    return cleandata


def medDisc(cleandata, varname):
    # weight of evidence encoding?
    # mean encoding?
    # https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
    # save mappings to apply to future data
    univals = np.sort(cleandata[[varname]].dropna()[varname].unique())
    arbindex = (range(len(univals)))
    valdict = dict(zip(univals, arbindex))
    cleandata[varname] = cleandata[varname].replace(valdict)
    session["medDisc"].append([varname,univals])
    return cleandata


def smallDisc(cleandata, varname):
    univals = np.sort(cleandata[[varname]].dropna()[varname].unique())
    dummies = pd.get_dummies(cleandata[varname])
    dummies = dummies.sort_index(axis = 1)
    dummies.columns = [varname + '_' + str(col) for col in dummies.columns]
    cleandata = pd.concat([cleandata, dummies], axis=1)
    cleandata = cleandata.drop([varname], axis=1)
    session["smallDisc"].append([varname,univals])
    return cleandata


def cleanTarget(cleandata, targetvar):
    if(cleandata[targetvar].dtype != np.float64 and cleandata[targetvar].dtype != np.int64):
        try:
            cleandata[targetvar] = pd.to_numeric(cleandata[targetvar])
        except Exception:
            cleandata = targetMap(cleandata, targetvar)
    return cleandata


def targetMap(cleandata, targetvar):
    univals = np.sort(cleandata[[targetvar]].dropna()[targetvar].unique())
    arbindex = (range(len(univals)))
    valdict = dict(zip(univals, arbindex))
    cleandata[targetvar] = cleandata[targetvar].replace(valdict)
    session["tarDisc"] = [[targetvar,valdict]]
    return cleandata


def revealClean():
    cleanedvars = []
    for k in session["datetimes"]:
        cleanedvars.append(str(k) + ' was converted into [' + str(k) + '_year, ' + str(k) + '_month, ' + str(k) + '_day].')
    for k in session["smallDisc"]:
        cleanedvars.append(str(k[0]) + ' was converted into ' + str(k[1]).replace("'", "").replace(" ", ", ") + '.')
    for k in session["medDisc"]:
        cleanedvars.append(str(k[0]) + ' was converted into a mapping of numbers.')
    for k in session["bigDisc"]:
        cleanedvars.append(str(k) + ' was dropped for being uninformative.')
    for k in session["nullvar"]:
        cleanedvars.append(str(k) + ' was dropped for having too many nulls.')
    for k in session["tarDisc"]:
        cleanedvars.append(str(k[0]) + ' was mapped on to: ' + str(k[1]))
    return cleanedvars


def cleanFuture(futuredata):
    cleandata = futuredata.copy()
    # replace with method calls
    for k in session["datetimes"]:
        cleandata[k] =  pd.to_datetime(cleandata[k], errors='coerce')
        cleandata[k + '_year'] = cleandata[k].dt.year
        cleandata[k + '_month'] = cleandata[k].dt.month
        cleandata[k + '_day'] = cleandata[k].dt.day
        cleandata = cleandata.drop([k], axis=1)
    for k in session["smallDisc"]:
        for dum in k[1]:
            newvar = k[0] + '_' + str(dum)
            cleandata[newvar] = 0
            cleandata[newvar] = np.where((cleandata[k[0]] == dum), 1, 0)
        cleandata = cleandata.drop([k[0]], axis=1)
    for k in session["medDisc"]:
        univals = k[1]
        arbindex = (range(len(univals)))
        cleandata[k] = cleandata[k].replace(valdict)
    for k in session["bigDisc"]:
        cleandata = cleandata.drop([k], axis=1)
    for k in session["nullvar"]:
        cleandata = cleandata.drop([k], axis=1)
    cleandata = cleandata.dropna()
    cleandata = cleandata.sort_index(axis = 1) 
    return cleandata


def normalize(cleandata):
    normdata = cleandata.copy()
    allcols = list(normdata.columns)
    allcols = allcols[:-1]
    for col in allcols:
        normdata[col] = (normdata[col] - normdata[col].mean())/normdata[col].std(ddof=0)
    normdata = normdata.dropna(axis='columns')
    session["normdata"] = normdata
    return normdata


def makeDists(cleandata, varname):
    vardata = cleandata[varname].values
    data = [go.Histogram(x=vardata)]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route('/fullhistchange', methods=['GET', 'POST'])
def fullChangeDist():
    varname = request.args['histchoice']
    cleandata = session["fulldata"]
    graphJSON = makeDists(cleandata, varname)
    return graphJSON


@app.route('/cleanhistchange', methods=['GET', 'POST'])
def cleanChangeDist():
    varname = request.args['histchoice']
    cleandata = session["normdata"]
    graphJSON = makeDists(cleandata, varname)
    return graphJSON
    # Split into files later ((html, css, js),(python, flask))


def rfeAlgo(cleandata):
    targetvar = cleandata.columns.values[-1]
    output = cleandata[targetvar].values
    inputs = cleandata.drop([targetvar], axis=1)
    rfemodel = LinearRegression()
    rfe = RFE(rfemodel, 1)
    fitrfe = rfe.fit(inputs.values, output)
    ranks = pd.DataFrame((inputs.columns.values), columns=['Feature'])
    rferanks = fitrfe.ranking_
    ranks['Rank'] = rferanks
    ranks = ranks.sort_values('Rank', ascending=True)
    ranks = ranks.set_index('Rank')
    # replace with ranks.index = ranks['Rank'].values?
    ranks = ranks.head(15).to_html(classes='rferank')
    # add corr matrix results?
    return ranks


def corrMatrix(cleandata):
    heatmap = cleandata.corr().iloc[::-1].values
    labels = cleandata.columns.values
    data = [go.Heatmap(z=heatmap, x=labels, y=np.flip(labels), colorscale='RdBu', reversescale=True, showscale=True, zmax=1, zmin=-1)]
    # use bluescale and absolute value here?
    # drop above diagonal?
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def splitTrainTest(cleandata, line):
    train = cleandata[0:(round(len(cleandata)*line))] 
    test = cleandata[(round(len(cleandata)*line)):]
    targetvar = cleandata.columns.values[-1]
    trainindex = train.index.values
    testindex = test.index.values
    trainoutput = train[targetvar].values
    testoutput = test[targetvar].values
    traininput = train.drop([targetvar], axis=1).values
    testinput = test.drop([targetvar], axis=1).values
    splitdata = [trainindex, testindex, traininput, testinput, trainoutput, testoutput]
    return splitdata


def linearRegression(splitdata):
    trainindex = splitdata[0]
    testindex = splitdata[1]
    traininput = splitdata[2]
    testinput = splitdata[3]
    trainoutput = splitdata[4]
    testoutput = splitdata[5]
    model = LinearRegression().fit(traininput, trainoutput)
    predictedtrain = model.predict(traininput)
    predictedtest = model.predict(testinput)
    results = [trainindex, testindex, trainoutput, predictedtrain, testoutput, predictedtest]
    session["LinRegResults"] = results
    return results


def kNearest(splitdata):
    trainindex = splitdata[0]
    testindex = splitdata[1]
    traininput = splitdata[2]
    testinput = splitdata[3]
    trainoutput = splitdata[4]
    testoutput = splitdata[5]
    model = KNeighborsRegressor().fit(traininput, trainoutput)
    predictedtrain = model.predict(traininput)
    predictedtest = model.predict(testinput)
    results = [trainindex, testindex, trainoutput, predictedtrain, testoutput, predictedtest]
    session["KNNResults"] = results
    return results


def extraTrees(splitdata):
    trainindex = splitdata[0]
    testindex = splitdata[1]
    traininput = splitdata[2]
    testinput = splitdata[3]
    trainoutput = splitdata[4]
    testoutput = splitdata[5]
    model = ExtraTreesRegressor(n_estimators=100).fit(traininput, trainoutput)
    predictedtrain = model.predict(traininput)
    predictedtest = model.predict(testinput)
    results = [trainindex, testindex, trainoutput, predictedtrain, testoutput, predictedtest]
    session["ExTreeResults"] = results
    return results


def xgBoost(splitdata):
    trainindex = splitdata[0]
    testindex = splitdata[1]
    traininput = splitdata[2]
    testinput = splitdata[3]
    trainoutput = splitdata[4]
    testoutput = splitdata[5]
    model = XGBRegressor(objective='reg:squarederror').fit(traininput, trainoutput)
    predictedtrain = model.predict(traininput)
    predictedtest = model.predict(testinput)
    results = [trainindex, testindex, trainoutput, predictedtrain, testoutput, predictedtest]
    session["XGBoostResults"] = results
    return results


def graphModelResults(results, split):
    trainindex = results[0]
    testindex = results[1]
    trainoutput = results[2]
    predictedtrain = results[3]
    testoutput = results[4]
    predictedtest = results[5]
    if split == 'Training':
        data = [go.Scatter(x=trainindex, y=trainoutput, name='Ground Truth', mode='lines'), go.Scatter(x=trainindex, y=predictedtrain, name='Prediction', mode='lines')]
        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        return graphJSON
    else:
        data = [go.Scatter(x=testindex, y=testoutput, name='Ground Truth', mode='lines'), go.Scatter(x=testindex, y=predictedtest, name='Prediction', mode='lines')]
        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        return graphJSON


@app.route('/splitchange', methods=['GET', 'POST'])
def changesplit():
    split = request.args['splitchoice']
    model = request.args['algorithm']
    if model == 'Linear Regression':
        results = session["LinRegResults"]
    elif model == 'K-Nearest Neighbors':
        results = session["KNNResults"]
    elif model == 'Extra Trees':
        results = session["ExTreeResults"]
    elif model == 'XGBoost':
        results = session["XGBoostResults"]
    else:
        results = session["LinRegResults"]
    graphJSON = graphModelResults(results, split)
    return graphJSON


def adj_r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))


def metricModelResults(results, cleandata):
    # return result table and test array
    tabewidth = len(cleandata.columns.values) - 1
    trainoutput = results[2]
    predictedtrain = results[3]
    testoutput = results[4]
    predictedtest = results[5]
    trainsize = len(trainoutput)
    testsize = len(testoutput)
    trainr2 = round(r2_score(trainoutput, predictedtrain), 4)
    testr2 = round(r2_score(testoutput, predictedtest), 4)
    trainadjr2 = round(adj_r2_score(trainr2, trainsize, tabewidth), 4)
    testadjr2 = round(adj_r2_score(testr2, testsize, tabewidth), 4)
    trainrmse = round(np.sqrt(mean_squared_error(trainoutput, predictedtrain)), 4)
    testrmse = round(np.sqrt(mean_squared_error(testoutput, predictedtest)), 4)
    trainmae = round(mean_absolute_error(trainoutput, predictedtrain), 4)
    testmae = round(mean_absolute_error(testoutput, predictedtest), 4)
    trainvar = round(explained_variance_score(trainoutput, predictedtrain), 4)
    testvar = round(explained_variance_score(testoutput, predictedtest), 4)
    trainarray = [trainr2, trainadjr2, trainrmse, trainmae, trainvar]
    testarray = [testr2, testadjr2, testrmse, testmae, testvar]
    resultdf = pd.DataFrame([trainarray, testarray],columns=['r2', 'adj-r2', 'rmse', 'mae', 'variance'])
    resultdf.index = ['Train', 'Test']
    resulttable = resultdf.to_html(classes='resulttable')
    metrics = [testarray, resulttable]
    return metrics


def fullSummary(linregresults, ranforresults, extreeresults, xgboostresults):
    summarydf = pd.DataFrame([linregresults, ranforresults, extreeresults, xgboostresults],columns=['r2', 'adj-r2', 'rmse', 'mae', 'variance'])
    algos = ['Linear Regression', 'K-Nearest Neighbors', 'Extra Trees', 'XGBoost']
    summarydf.index = algos
    summary = summarydf.to_html(classes='finaltable')
    bestalgo = algos[np.where(summarydf['r2'].values == np.amax(summarydf['r2'].values))[0][0]]
    results = [summary, bestalgo]
    return results


def predictData(cleandata, bestalgo):
    testinput = cleandata.values
    traindata = session["normdata"]
    targetvar = traindata.columns.values[-1]
    trainoutput = traindata[targetvar].values
    traininput = traindata.drop([targetvar], axis=1).values
    if bestalgo == 'Linear Regression':
        model = LinearRegression().fit(traininput, trainoutput)
        results = model.predict(testinput)
    elif bestalgo == 'K-Nearest Neighbors':
        model = KNeighborsRegressor().fit(traininput, trainoutput)
        results = model.predict(testinput)
    elif bestalgo == 'Extra Trees':
        model = ExtraTreesRegressor(n_estimators=100).fit(traininput, trainoutput)
        results = model.predict(testinput)
    elif bestalgo == 'XGBoost':
        model = XGBRegressor(objective='reg:squarederror').fit(traininput, trainoutput)
        results = model.predict(testinput)
    else:
        model = LinearRegression().fit(traininput, trainoutput)
        results = model.predict(testinput)
    cleandata[targetvar] = results
    predictions = cleandata
    return predictions


def saveData(future, predictions, bestalgo):
    filepath = future[:-4]
    filepath = filepath + '_' + bestalgo + '_results.csv'
    predictions.to_csv(filepath) 
    return ''


@socketio.on('disconnect')
def disconnect_user():
    flask.ext.login.logout_user()
    clearCache()
    session.clear()


if __name__ == '__main__':
    # write comments for everything later
    app.debug = True
    app.run()