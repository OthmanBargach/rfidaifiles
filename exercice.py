
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

    # Clear the screen
os.system("cls") # For Windows

    # files
moving_file = './data/moving.csv'
stationary_file = './data/stationary.csv'
tags_file = './data/ko=2__tiltAngle=40.csv'

    # Reads moving data
moving_data = pd.read_csv(moving_file, names=['EPC'])
moving_data['actual'] = 'moving'

    # Reads stationary data
stationary_data = pd.read_csv(stationary_file, names=['EPC'])
stationary_data['actual'] = 'stationary'

    # Group by EPC and Antenna
reflist = pd.concat([moving_data, stationary_data], ignore_index=True)

    # Reads the data and skip the 3 first lines
cols = ['Timestamp', 'EPC', 'TID', 'Antenna', 'RSSI', 'Frequency', 'Hostname', 'PhaseAngle', 'DopplerFrequency']
tags = pd.read_csv(tags_file, skiprows=3, sep=';', names=cols)
tags = tags[['Timestamp', 'EPC', 'Antenna', 'RSSI']]


    # Convert the Timestamp to datetime
tags['Timestamp'] = pd.to_datetime(tags['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

    # Convert the RSSI to float and remove the , by .
tags['RSSI'] = tags['RSSI'].str.replace(',', '.').astype(float)

    # Add orientation column with antenna 1 and 2 to in and 3 and 4 to out
dic_Antenna_coverage = {1: 'in', 2: 'in', 3: 'out', 4: 'out'}
tags['Antenna_coverage'] = tags['Antenna'].map(dic_Antenna_coverage)

    # Add column with moving or stationnary
tags = pd.merge(tags, reflist, on='EPC', how='left')

    # Group by epc and antenna
RSSImax = tags.groupby('EPC') ['RSSI'].max().rename('RSSI_max').reset_index()
RSSImax = pd.merge(RSSImax, reflist, on='EPC', how='left')


    # Plot the data
sns.set(style="whitegrid")
ax = sns.boxplot(x="actual", y="RSSI_max", data=RSSImax)
ax = sns.swarmplot(x="actual", y="RSSI_max", data=RSSImax, color=".25")
ax.set_title('RSSI max by orientation')
ax.set_xlabel('Orientation')
ax.set_ylabel('RSSI max')
ax.figure.savefig('./results/RSSI_max.png')

    # Separate the data into two groups (moving and stationary) by prediction on RSSI_max = -68
RSSImax['predicted'] = RSSImax['RSSI_max'].apply(lambda x: 'moving' if x > -68 else 'stationary')

    # Confusion matrix 
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(RSSImax['actual'], RSSImax['predicted'])

    # Plot the confusion matrix
disp = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
disp.set_title('Confusion matrix')
disp.set_xlabel('Predicted')
disp.set_ylabel('Actual')
disp.figure.savefig('./results/confusion_matrix.png')

    # the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(RSSImax['actual'], RSSImax['predicted'])

    # Classification report and saves it in a file
Report = classification_report(RSSImax['actual'], RSSImax['predicted'])
with open('./results/classification_report.txt', 'w') as f:
    f.write(Report)


    # Plot the classification report
fig, ax = plt.subplots(figsize=(10, 10))
ax.text(0.05, 0.95, Report, fontsize=14, va='top')
ax.axis('off')
fig.savefig('./results/classification_report.png')

    # Plot the data by time and RSSI
plt.figure(figsize=(14,8))
Tmin=tags['Timestamp'].min()
Tmax=tags['Timestamp'].max()
dict_Antenna_coverage = {'in':'blue', 'out':'red'}
dict_actual = {'moving':'o', 'stationary':'+'}
for key, df in tags.groupby(['actual', 'Antenna_coverage']):
    actual=key[0]
    Antenna_coverage=key[1]
    m=dict_actual[actual]
    c=dict_Antenna_coverage[Antenna_coverage]
    sns.scatterplot(data=df, x='Timestamp', y='RSSI', marker=m, color=c)
plt.xlim(Tmin, Tmax)
plt.title('RSSI by time')
plt.xlabel('Time')
plt.ylabel('RSSI')
plt.savefig('./results/RSSI_time.png')

