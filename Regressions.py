from flask import Flask, render_template, request, session
from flask_session import Session
import numpy as np
import pandas as pd
import json
from os import path
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import plotly
import plotly.graph_objs as go


app = Flask(__name__)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)


@app.route('/')
def index():
    return render_template('my-form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    maindata = request.form['maindata']
    future = request.form['future']
    session.clear()
    checkflag = validate(maindata)
    futureflag = validate(future)
    if checkflag == True:
        results = fullAnalysis(maindata)
        allvars = session["fulldata"].columns.values
        cleanvars = session["cleandata"].columns.values
        return render_template('index.html', sumtables=results[0], howcleanedvars=results[1], allvars=allvars, cleanvars=cleanvars, rawdistplot=results[2], cleandistplot=results[3], heatmap=results[4], rfelist=results[5], linreggraph=results[6], linregtable=results[7], ranforgraph=results[8], ranfortable=results[9], extreegraph=results[10], extreetable=results[11], xgboostgraph=results[12], xgboosttable=results[13])
    else:
        return render_template('error-page.html')
    # if futureflag == True:
    #    pred = predict(futureflag)
    #    in predict check if columns are named the same as in train
    #    error handling to make sure data names and internals are same


def validate(maindata):
    checkflag = False
    validpath = path.exists(maindata)
    if validpath:
        try:
            df1 = pd.read_csv(maindata) 
            checkflag = True
        except Exception:
            checkflag = False
    return checkflag


def fullAnalysis(maindata):
    fulldata = readfile(maindata)
    summary = summarize(fulldata)
    cleandata = cleanColumns(fulldata)
    # outdata = outliers(cleandata) - optional <remove outliers>
    howcleanedvars = revealClean()
    rawdistplot = makeDists(fulldata, fulldata.columns.values[0])
    cleandistplot = makeDists(cleandata, cleandata.columns.values[0])
    # normdata = normalize(cleandata) - optional <normalization>
    heatmap = corrMatrix(cleandata)
    rfelist = rfeAlgo(cleandata)
    splitdata = splitTrainTest(cleandata, .8)
    linreg = linearRegression(splitdata)
    linreggraph = graphModelResults(linreg, 'Training')
    linregresults = metricModelResults(linreg, cleandata)
    linregtable = linregresults[1]
    ranfor = randomForest(splitdata)
    ranforgraph = graphModelResults(ranfor, 'Training')
    ranforresults = metricModelResults(ranfor, cleandata)
    ranfortable = ranforresults[1]
    extree = extraTrees(splitdata)
    extreegraph = graphModelResults(extree, 'Training')
    extreeresults = metricModelResults(extree, cleandata)
    extreetable = extreeresults[1]
    xgboost = xgBoost(splitdata)
    xgboostgraph = graphModelResults(xgboost, 'Training')
    xgboostresults = metricModelResults(xgboost, cleandata)
    xgboosttable = xgboostresults[1]
    results = [summary, howcleanedvars, rawdistplot, cleandistplot, heatmap, rfelist, linreggraph, linregtable, ranforgraph, ranfortable, extreegraph, extreetable, xgboostgraph, xgboosttable]
    return results
    # return array of all visuals needed


def readfile(maindata):
	# clean string for common errors
	fulldata = pd.read_csv(maindata)
	# check file type and read diff types
	if (fulldata.columns.values[0] == 'Unnamed: 0'):
		fulldata = fulldata.drop(['Unnamed: 0'], axis=1)
    # add logic to check if col 1 is an index then fulldata = fulldata.set_index(fulldata.columns.values[0])?
	return fulldata


def summarize(fulldata):
    # Clean up and extract commons later
    # Add names for boxes
    summary =  fulldata.describe(include = 'all').T
    sumlabels = summary.columns.values
    if ('unique' in sumlabels) & ('mean' in sumlabels):
        summaries = mixsummarize(fulldata)
    elif ('unique' in sumlabels):
        summaries = discsummarize(fulldata)
    else:
        summaries = contsummarize(fulldata)
    return summaries


def contsummarize(fulldata):
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


def discsummarize(fulldata):
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


def mixsummarize(fulldata):
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
	cleandata = fulldata.dropna()
	cleandata = cleandata.sort_index(axis = 0) 
	varnames = cleandata.columns.values
	targetvar = varnames[-1]
	targetdata = cleandata[targetvar].values
	cleandata = cleandata.drop([targetvar], axis=1)
	nacols = cleandata.isna().sum().values
	for k in range(len(nacols)):
		if (nacols[k]/len(cleandata) > .1):
			dropvar = varnames[k]
			cleandata = cleandata.drop([dropvar], axis=1)
	session["datetimes"] = []
	session["smallDisc"] = []
	session["medDisc"] = []
	session["bigDisc"] = []
	for varname in cleandata.columns.values:
	    if(cleandata[varname].dtype != np.float64 and cleandata[varname].dtype != np.int64):
	    	try:
	    		cleandata[varname] =  pd.to_numeric(cleandata[varname])
	    	except Exception:
	    		cleandata = discClean(cleandata, varname)
	# check for numbers (convert all to numeric at end to be safe)?
	cleandata = cleandata.sort_index(axis = 1) 
	cleandata[targetvar] = targetdata
	cleandata = cleandata.dropna()
	session["cleandata"] = cleandata
	return cleandata
    # add clean colums with appending train and prod


def discClean(cleandata, varname):
	# do other checks to ensure data is a datetime
	try:
	    cleandata[varname] =  pd.to_datetime(cleandata[varname])
	    cleandata = datetimeDisc(cleandata, varname)
	except Exception:
		# instead of size, change it to nominal vs ordinal
	    univals = len(cleandata[varname].unique())
	    if (univals < 2) | (univals/len(cleandata)>.1):
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
	session["bigDisc"].append(varname)
	return cleandata


def medDisc(cleandata, varname):
	# weight of evidence encoding?
	# mean encoding?
	# https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
	# save mappings to apply to future data
	univals = np.sort(cleandata[varname].unique())
	arbindex = (range(len(univals)))
	valdict = dict(zip(univals, arbindex))
	cleandata[varname] = cleandata[varname].replace(valdict)
	session["medDisc"].append([varname,univals])
	return cleandata


def smallDisc(cleandata, varname):
	univals = np.sort(cleandata[varname].unique())
	dummies = pd.get_dummies(cleandata[varname])
	dummies = dummies.sort_index(axis = 1)
	dummies.columns = [varname + '_' + str(col) for col in dummies.columns]
	cleandata = pd.concat([cleandata, dummies], axis=1)
	cleandata = cleandata.drop([varname], axis=1)
	session["smallDisc"].append([varname,univals])
	# change this to work with testing
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
		cleanedvars.append(str(k) + ' was dropped.')
	return cleanedvars


def makeDists(cleandata, varname):
	vardata = cleandata[varname].values
	data = [go.Histogram(x=vardata)]
	graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON


@app.route('/fullhistchange', methods=['GET', 'POST'])
def fullchangedist():
    varname = request.args['histchoice']
    cleandata = session["fulldata"]
    graphJSON = makeDists(cleandata, varname)
    return graphJSON


@app.route('/cleanhistchange', methods=['GET', 'POST'])
def cleanchangedist():
    varname = request.args['histchoice']
    cleandata = session["cleandata"]
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
	ranks = ranks.head(20).to_html(classes='rferank')
	# test bigger data and change 15 to max stretch
	# add corr matrix results?
	return ranks


def corrMatrix(cleandata):
	heatmap = cleandata.corr().iloc[::-1].values
	labels = cleandata.columns.values
	data = [go.Heatmap(z=heatmap, x=labels, y=np.flip(labels), colorscale='RdBu', reversescale=True, showscale=False)]
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


def randomForest(splitdata):
	trainindex = splitdata[0]
	testindex = splitdata[1]
	traininput = splitdata[2]
	testinput = splitdata[3]
	trainoutput = splitdata[4]
	testoutput = splitdata[5]
	model = RandomForestRegressor().fit(traininput, trainoutput)
	predictedtrain = model.predict(traininput)
	predictedtest = model.predict(testinput)
	results = [trainindex, testindex, trainoutput, predictedtrain, testoutput, predictedtest]
	session["RanForResults"] = results
	return results


def extraTrees(splitdata):
	trainindex = splitdata[0]
	testindex = splitdata[1]
	traininput = splitdata[2]
	testinput = splitdata[3]
	trainoutput = splitdata[4]
	testoutput = splitdata[5]
	model = ExtraTreesRegressor().fit(traininput, trainoutput)
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
	model = XGBRegressor().fit(traininput, trainoutput)
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
		data = [go.Scatter(x=trainindex, y=trainoutput, name='Ground Truth'), go.Scatter(x=trainindex, y=predictedtrain, name='Prediction')]
		graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
		return graphJSON
	else:
		data = [go.Scatter(x=testindex, y=testoutput, name='Ground Truth'), go.Scatter(x=testindex, y=predictedtest, name='Prediction')]
		graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
		return graphJSON


@app.route('/splitchange', methods=['GET', 'POST'])
def changesplit():
    split = request.args['splitchoice']
    model = request.args['algorithm']
    if model == 'Linear Regression':
    	results = session["LinRegResults"]
    elif model == 'Random Forrest':
    	results = session["RanForResults"]
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
	trainrmse = round(mean_squared_error(trainoutput, predictedtrain), 4)
	testrmse = round(mean_squared_error(testoutput, predictedtest), 4)
	trainmae = round(mean_absolute_error(trainoutput, predictedtrain), 4)
	testmae = round(mean_absolute_error(testoutput, predictedtest), 4)
	trainvar = round(explained_variance_score(trainoutput, predictedtrain), 4)
	testvar = round(explained_variance_score(testoutput, predictedtest), 4)
	trainarray = [trainr2, trainadjr2, trainrmse, trainmae, trainvar]
	testarray = [testr2, testadjr2, testrmse, testmae, testvar]
	resultdf = pd.DataFrame([trainarray, testarray],columns=['r2', 'adj-r2', 'mse', 'mae', 'variance'])
	resultdf.index = ['Train', 'Test']
	resulttable = resultdf.to_html(classes='resulttable')
	metrics = [testarray, resulttable]
	return metrics


if __name__ == '__main__':
	# write comments for everything later
    app.debug = True
    app.run()