from flask import Flask, render_template, request, session
from flask_session import Session
import plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
from os import path


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
        return render_template('index.html', sumtables=results[0], howcleanedvars=results[1], allvars=allvars, cleanvars=cleanvars, rawdistplot=results[2], cleandistplot=results[3], heatmap=results[4])
    else:
        return render_template('error-page.html')
        # replace this with a stylized error pade like github
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
    cleanedvars = revealClean()
    rawdistplot = makeDists(fulldata, fulldata.columns.values[0])
    cleandistplot = makeDists(cleandata, cleandata.columns.values[0])
    # rfelist = rfeAlgo(cleandata)
    heatmap = corrMatrix(cleandata)
    # outdata = outliers(cleandata) - optional <remove outliers>
    # normdata = normalize(cleandata) - optional <normalization>
    results = [summary, cleanedvars, rawdistplot, cleandistplot, heatmap]
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
    # IMPORTANT
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
	session["fulldata"] = fulldata
	# drop cols with more than 25% na
	cleandata = fulldata.dropna()
	cleandata = cleandata.sort_index(axis = 0) 
	targetvar = cleandata.columns.values[-1]
	targetdata = cleandata[targetvar].values
	cleandata = cleandata.drop([targetvar], axis=1)
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
	cleandata = cleandata.dropna()
	# check for numbers (convert all to numeric at end to be safe)?
	cleandata = cleandata.sort_index(axis = 1) 
	cleandata[targetvar] = targetdata
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
	return ''


def corrMatrix(cleandata):
	heatmap = cleandata.corr().iloc[::-1].values
	labels = cleandata.columns.values
	data = [go.Heatmap(z=heatmap, x=labels, y=np.flip(labels), colorscale='RdBu', reversescale=True)]
	# use bluescale and absolute value here?
	# drop above diagonal?
	graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON


if __name__ == '__main__':
    app.debug = True
    app.run()