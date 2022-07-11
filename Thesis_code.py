# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:16:01 2022

@author: Calvin Paperwala
"""

# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels
import pyrankvote
from pyrankvote import Candidate, Ballot
from scipy import interpolate

# Import datasets
data_to2018 = pd.read_excel(r'C:\Users\calvi\Data\University\Thesis\Data\eurovision_song_contest_1975_2019.xlsx')
df = data_to2018

data_2019_2021 = pd.read_excel(r'C:\Users\calvi\Data\University\Thesis\Data\EV_2019-2021.xlsx')

# Phase 1: Data Cleaning of 1975-2019 dataset

# Remove space from columns
df.columns = df.columns.str.rstrip()

# Filter only for 2016-2018
df = df[df['Year'] >= 2016]
df = df[df['Year'] <= 2018]

# Dataframe for alternative av
df_v2 = df[df['(semi-) final'] == 'f']
df_v2['Voter ID'] = df_v2['From country'] + "_" + df_v2['Jury or Televoting']


df_v2 = df_v2.pivot(index=['Year', 'Voter ID'], columns='To country', values='Points')
df_v2.to_csv("Results/df_vs.csv")

# Phase 2: Merging with 2019/2021 dataset

# Melt 2019/2021 dataset to resemble 1975-2019 data
df2 = pd.melt(data_2019_2021, id_vars=['Year', '(semi-) final', 'Edition', 'Jury or Televoting', 'To country'],
              value_vars=['Albania', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium', 'Bulgaria',
                          'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia',
                          'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Israel', 'Italy', 'Latvia',
                          'Lithuania', 'Malta', 'Moldova', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway',
                          'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovenia', 'Spain',
                          'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom'],
              var_name='From country', value_name='Points')


# Rearrange order of columns
df2 = df2[['Year', '(semi-) final', 'Edition', 'Jury or Televoting', 'From country', 'To country', 'Points']]

# Stack the two datasets
df = pd.concat([df, df2])

# Replace NA with 0 in Points Column
df['Points'] = df['Points'].fillna(0)

# Phase 3: Prepare data for analysis

# Filter only for finals
df = df[df['(semi-) final'] == 'f']

# Plurality Vote Column
df['PV'] = np.where(df['Points'] == 12, 1, 0)

# Alternative Vote Column
df['AV'] = np.where(df['Points'] == 10, 9, df['Points'])
df['AV'] = np.where(df['Points'] == 12, 10, df['AV'])

# Borda Count Column
df['BC'] = df['AV']

# Coombs' Method Column
df['CM'] = df['AV']

# Approval Vote Column
df['ApV'] = np.where(df['Points'] == 0, 0, 1)

# Result Concat Columns
df['Points_Country'] = df['To country'] + "++" + df['Points'].map(str)

# Unique voter id
df['Voter_ID'] = df['From country'] + "_" + df['Jury or Televoting'] + "_" + df['Year'].map(str)

# Split data by year
df_2016 = df[df['Year'] == 2016]
df_2017 = df[df['Year'] == 2017]
df_2018 = df[df['Year'] == 2018]
df_2019 = df[df['Year'] == 2019]
df_2021 = df[df['Year'] == 2021]

# Phase 2: Data Analysis

# Define functions for Bootstrap Replication


def or_winner(data, country='To country', method='Points'):
    """"Determines the winner of the election based on original votes"""
    grouped_df = data.groupby(['Year', country])[method].sum().reset_index()
    original_results = grouped_df.groupby(['Year']).apply(
        lambda x: x.sort_values([method], ascending=False)).reset_index(drop=True)
    winner = original_results.groupby('Year').apply(lambda x: x.nlargest(1, method)).reset_index(drop=True)
    return winner.iloc[0][country]


def pv_winner(data, country='To country', method='PV'):
    """"Determines the winner of the election based on plurality vote"""
    data['PV'] = np.where(data['Points'] == 12, 1, 0)
    grouped_df = data.groupby(['Year', country])[method].sum().reset_index()
    original_results = grouped_df.groupby(['Year']).apply(
        lambda x: x.sort_values([method], ascending=False)).reset_index(drop=True)
    winner = original_results.groupby('Year').apply(lambda x: x.nlargest(1, method)).reset_index(drop=True)
    return winner.iloc[0][country]


def bc_winner(data, country='To country', method='BC'):
    """"Determines the winner of the election based on borda count"""
    data['BC'] = np.where(data['Points'] == 10, 9, data['Points'])
    data['BC'] = np.where(data['Points'] == 12, 10, data['BC'])
    grouped_df = data.groupby(['Year', country])[method].sum().reset_index()
    original_results = grouped_df.groupby(['Year']).apply(
        lambda x: x.sort_values([method], ascending=False)).reset_index(drop=True)
    winner = original_results.groupby('Year').apply(lambda x: x.nlargest(1, method)).reset_index(drop=True)
    return winner.iloc[0][country]


def av_winner(data, country='To country', method='AV'):
    """"Determines the winner of the election based on alternative vote"""
    data['AV'] = np.where(data['Points'] == 10, 9, data['Points'])
    data['AV'] = np.where(data['Points'] == 12, 10, data['AV'])
    grouped_av_df = data.groupby(['Year', country])[method].agg({('zero', lambda x: (x == 0).sum()),
                                                                 ('one', lambda x: (x == 1).sum()),
                                                                 ('two', lambda x: (x == 2).sum()),
                                                                 ('three', lambda x: (x == 3).sum()),
                                                                 ('four', lambda x: (x == 4).sum()),
                                                                 ('five', lambda x: (x == 5).sum()),
                                                                 ('six', lambda x: (x == 6).sum()),
                                                                 ('seven', lambda x: (x == 7).sum()),
                                                                 ('eight', lambda x: (x == 8).sum()),
                                                                 ('nine', lambda x: (x == 9).sum()),
                                                                 ('ten', lambda x: (x == 10).sum())}).reset_index()
    while sum(grouped_av_df['ten']/sum(grouped_av_df['ten']) > 0.5) == 0:
        del_list = grouped_av_df.index[grouped_av_df['ten'] == 0].tolist()
        grouped_av_df = grouped_av_df.drop(labels=del_list, axis=0)
        move_countries = grouped_av_df[grouped_av_df['ten'] == min(grouped_av_df['ten'])][country].unique().tolist()
        move_list = data.index[data['To country'].isin(move_countries) & (data[method] == 10)].tolist()
        data.loc[move_list, method] = 0
        voters = data.loc[move_list]['Voter_ID'].tolist()
        X = data[data['Voter_ID'].isin(voters)]
        change_list = X.index[X[method] == X[method].max()].tolist()
        data.loc[change_list, method] = 10
        grouped_av_df = data.groupby(['Year', country])[method].agg({('zero', lambda x: (x == 0).sum()),
                                                                     ('one', lambda x: (x == 1).sum()),
                                                                     ('two', lambda x: (x == 2).sum()),
                                                                     ('three', lambda x: (x == 3).sum()),
                                                                     ('four', lambda x: (x == 4).sum()),
                                                                     ('five', lambda x: (x == 5).sum()),
                                                                     ('six', lambda x: (x == 6).sum()),
                                                                     ('seven', lambda x: (x == 7).sum()),
                                                                     ('eight', lambda x: (x == 8).sum()),
                                                                     ('nine', lambda x: (x == 9).sum()),
                                                                     ('ten', lambda x: (x == 10).sum())}).reset_index()
    else:
        winner = grouped_av_df.groupby('Year').apply(lambda x: x.nlargest(1, 'ten')).reset_index(drop=True)
    return winner.iloc[0][country]


def cm_winner(data, country='To country', method='CM'):
    """"Determines the winner of the election based on Coombs' method"""
    data['CM'] = np.where(data['Points'] == 10, 9, data['Points'])
    data['CM'] = np.where(data['Points'] == 12, 10, data['CM'])
    grouped_cm_df = data.groupby(['Year', 'To country'])['CM'].agg({('zero', lambda x: (x == 0).sum()),
                                                                    ('one', lambda x: (x == 1).sum()),
                                                                    ('two', lambda x: (x == 2).sum()),
                                                                    ('three', lambda x: (x == 3).sum()),
                                                                    ('four', lambda x: (x == 4).sum()),
                                                                    ('five', lambda x: (x == 5).sum()),
                                                                    ('six', lambda x: (x == 6).sum()),
                                                                    ('seven', lambda x: (x == 7).sum()),
                                                                    ('eight', lambda x: (x == 8).sum()),
                                                                    ('nine', lambda x: (x == 9).sum()),
                                                                    ('ten', lambda x: (x == 10).sum())}).reset_index()
    while sum(grouped_cm_df['ten']/sum(grouped_cm_df['ten']) > 0.5) == 0:

        # Find the countries with most bottom votes
        if max(grouped_cm_df['zero']):
            del_list = grouped_cm_df[grouped_cm_df['zero'] == max(grouped_cm_df['zero'])]['To country'].unique().tolist()
        else:
            del_list = grouped_cm_df[grouped_cm_df['one'] == max(grouped_cm_df['one'])]['To country'].unique().tolist()
        # Find the voters who voted for this country as top country
        move_list = data.index[data['To country'].isin(del_list) & (data['CM'] == 10)].tolist()

        # Change their top vote to 0
        data.loc[move_list, 'CM'] = 0

        # Change their second vote to 10
        voters = data.loc[move_list]['Voter_ID'].tolist()
        X = data[data['Voter_ID'].isin(voters)]
        change_list = X.index[X['CM'] == X['CM'].max()].tolist()
        data.loc[change_list, 'CM'] = 10

        # Remove this country from the data
        data = data.drop(data[data['To country'].isin(del_list)].index)

        grouped_cm_df = data.groupby(['Year', 'To country'])['CM'].agg({('zero', lambda x: (x == 0).sum()),
                                                                        ('one', lambda x: (x == 1).sum()),
                                                                        ('two', lambda x: (x == 2).sum()),
                                                                        ('three', lambda x: (x == 3).sum()),
                                                                        ('four', lambda x: (x == 4).sum()),
                                                                        ('five', lambda x: (x == 5).sum()),
                                                                        ('six', lambda x: (x == 6).sum()),
                                                                        ('seven', lambda x: (x == 7).sum()),
                                                                        ('eight', lambda x: (x == 8).sum()),
                                                                        ('nine', lambda x: (x == 9).sum()),
                                                                        ('ten', lambda x: (x == 10).sum())}).reset_index()

    else:
        winner = grouped_cm_df.groupby('Year').apply(lambda x: x.nlargest(1, 'ten')).reset_index(drop=True)
    return winner.iloc[0][country]


def counter(x, data, col):
    return (data[col] == x).sum()


def cm_winner_v2(data=data_2019_2021, year=2021):

    # Transform data
    data = data_2019_2021[data_2019_2021['Year'] == 2021]
    j_data = data[data['Jury or Televoting'] == 'J']
    t_data = data[data['Jury or Televoting'] == 'T']


    j_data = j_data.iloc[:, 4:]
    j_data = j_data.rename(columns={'To country': 'Voter'})
    j_data.set_index('Voter', inplace=True)
    j_data_tr = j_data.transpose()
    j_data_tr.reset_index(inplace=True)
    j_data_tr = j_data_tr.rename(columns={'index': 'Voter ID'})
    j_data_tr['Voter ID'] = j_data_tr['Voter ID'] + '_j'

    t_data = t_data.iloc[:, 4:]
    t_data = t_data.rename(columns={'To country': 'Voter'})
    t_data.set_index('Voter', inplace=True)
    t_data_tr = t_data.transpose()
    t_data_tr.reset_index(inplace=True)
    t_data_tr = t_data_tr.rename(columns={'index': 'Voter ID'})
    t_data_tr['Voter ID'] = t_data_tr['Voter ID'] + '_t'

    cand_list = list(j_data_tr.columns)
    cand_list = sorted(cand_list)

    j_data_tr = j_data_tr[cand_list]
    t_data_tr = t_data_tr[cand_list]

    data_tr = pd.concat([j_data_tr, t_data_tr])

    # Convert to Coombs' method
    data_tr = data_tr.replace(10, 9)
    data_tr = data_tr.replace(12, 10)
    data_tr = data_tr.fillna(0)

    data_tr.to_csv("Results/Temp_2021_ordered.csv")


    # Find the country with the most bottom votes
    keys_list = []
    values_list = []

    for col in data_tr:
        keys_list.append(col)
        values_list.append(counter(0, data_tr, col=col))

    zip_iterator = zip(keys_list, values_list)
    zero_count = pd.DataFrame(zip_iterator)
    countries = zero_count.iloc[zero_count[1].idxmax()]

    # Change votes for these countries to 0

    for country in countries:
        data_tr[country]


def apv_winner(data, country='To country', method='ApV'):
    """"Determines the winner of the election based on approval vote"""
    data['ApV'] = np.where(data['Points'] == 0, 0, 1)
    grouped_df = data.groupby(['Year', country])[method].sum().reset_index()
    original_results = grouped_df.groupby(['Year']).apply(
        lambda x: x.sort_values([method], ascending=False)).reset_index(drop=True)
    winner = original_results.groupby('Year').apply(lambda x: x.nlargest(1, method)).reset_index(drop=True)
    return winner.iloc[0][country]


def bootstrap_replicates(data, func, replicates=2, random_state=21):
    winner_list = []
    temp = data.pivot(index='Voter_ID', columns='To country', values='Points')
    np.random.seed(random_state)
    for i in range(replicates):
        temp2 = pd.DataFrame(np.random.choice(data['Voter_ID'], size=len(temp)))
        temp2.columns = ['Voter_ID']
        temp2 = pd.merge(temp2, temp, on=['Voter_ID'])
        country_names = [col for col in temp2]
        country_names = country_names[1:]
        temp3 = pd.melt(temp2, id_vars=['Voter_ID'], value_vars=country_names,
                        var_name='To country', value_name='Points')
        temp3[['From country', 'Jury or Televoting', 'Year']] = temp3['Voter_ID'].str.split('_', 2, expand=True)
        winner_list.append(func(temp3))
    winner_list = pd.DataFrame(winner_list)
    return winner_list


def bootstrap_replicates_pr(ballots, func=pyrankvote.instant_runoff_voting, replicates=2, random_state=21):
    winner_list = []
    np.random.seed(random_state)
    for i in range(replicates):
        br_ballot = np.random.choice(ballots, size=len(ballots))
        election_result = func(candidates, br_ballot)
        winner_list.append(election_result.get_winners())
    winner_list = pd.DataFrame(winner_list)
    return winner_list


# Descriptive Statistics
jury_df = df[df['Jury or Televoting'] == 'J']
tv_df = df[df['Jury or Televoting'] == 'T']

total_votes_ds = pd.crosstab(df['To country'], df['Year'], values=df['Points'], aggfunc='sum')
jury_votes_ds = pd.crosstab(jury_df['To country'], jury_df['Year'], values=jury_df['Points'], aggfunc='sum')
tv_votes_ds = pd.crosstab(tv_df['To country'], tv_df['Year'], values=tv_df['Points'], aggfunc='sum')

total_votes_ds.to_excel("Results/Descriptive Statistics (Total Votes).xlsx")
jury_votes_ds.to_excel("Results/Descriptive Statistics (Jury Votes).xlsx")
tv_votes_ds.to_excel("Results/Descriptive Statistics (Tele Votes).xlsx")


# Candidate Type Analysis

grouped_vt_df = df.groupby(['Year', 'To country'])['Points'].\
                    agg({('Top 3', lambda x: ((x == 12) | (x == 10) | (x == 8)).sum()),
                         ('Bottom 3', lambda x: ((x == 0) | (x == 1) | (x == 2)).sum()),
                         ('Other', lambda x: ((x == 3) | (x == 4) | (x == 5) | (x == 6) | (x == 7)).sum())}).\
                    reset_index()

grouped_vt_df['Total votes received'] = grouped_vt_df['Top 3'] + grouped_vt_df['Bottom 3'] + grouped_vt_df['Other']
grouped_vt_df['Top3_%'] = grouped_vt_df['Top 3']/grouped_vt_df['Total votes received']*100
grouped_vt_df['Bottom3_%'] = grouped_vt_df['Bottom 3']/grouped_vt_df['Total votes received']*100

grouped_vt_df['Voter Type'] = np.where((grouped_vt_df['Top3_%'] > 20) & (grouped_vt_df['Bottom3_%'] > 20),
                                       "Polarizing", 0)
grouped_vt_df['Voter Type'] = np.where((grouped_vt_df['Top3_%'] > 20) & (grouped_vt_df['Bottom3_%'] <= 20),
                                       "Popular", grouped_vt_df['Voter Type'])
grouped_vt_df['Voter Type'] = np.where((grouped_vt_df['Top3_%'] <= 20) & (grouped_vt_df['Bottom3_%'] > 20),
                                       "Unpopular", grouped_vt_df['Voter Type'])
grouped_vt_df['Voter Type'] = np.where((grouped_vt_df['Top3_%'] <= 20) & (grouped_vt_df['Bottom3_%'] <= 20),
                                       "Medium", grouped_vt_df['Voter Type'])

Voter_type_year = pd.crosstab(grouped_vt_df['To country'], grouped_vt_df['Year'],
                              values=grouped_vt_df['Voter Type'], aggfunc='max')

Voter_type_year.to_csv("Results/Voter_type_year(20).csv")

grouped_vt_df_2016 = grouped_vt_df[grouped_vt_df['Year'] == 2016]
grouped_vt_df_2017 = grouped_vt_df[grouped_vt_df['Year'] == 2017].reset_index()
grouped_vt_df_2018 = grouped_vt_df[grouped_vt_df['Year'] == 2018].reset_index()
grouped_vt_df_2019 = grouped_vt_df[grouped_vt_df['Year'] == 2019].reset_index()
grouped_vt_df_2021 = grouped_vt_df[grouped_vt_df['Year'] == 2021].reset_index()


fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(15)
v = sns.scatterplot(data=grouped_vt_df_2021, x='Bottom3_%', y='Top3_%', ci=None)
v.set(xlabel="Bottom 3 Countries (%)",
      ylabel="Top 3 Countries (%)",
      title='Types of Candidates in 2021 (Ordinal Ranking)')
for i in range(grouped_vt_df_2021.shape[0]):
    plt.text(x=grouped_vt_df_2021['Bottom3_%'][i]+0.3, y=grouped_vt_df_2021['Top3_%'][i]+0.3,
             s=grouped_vt_df_2021['To country'][i], fontdict=dict(color='black', size=10))
plt.axvline(25, 0, 100, color='Black')
plt.axhline(25, 0, 100, color='Black')
plt.ylim(0, 50)
plt.xlim(0, 100)
plt.text(1, 25+1, "25% Line", horizontalalignment='left', size='large', color='black')
plt.text(25+1, 48, "25% Line", horizontalalignment='left', size='large', color='black')
plt.show()


fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(15)
v = sns.scatterplot()
v.set(xlabel="Bottom 3 Countries (%)",
      ylabel="Top 3 Countries (%)",
      title='Candidate Type Classification')
plt.ylim(0, 50)
plt.xlim(0, 50)
plt.text(1, 25+1, "25% Line", horizontalalignment='left', size='large', color='black')
plt.text(25+1, 48, "25% Line", horizontalalignment='left', size='large', color='black')
plt.axvline(25, 0, 100, color='Black')
plt.axhline(25, 0, 100, color='Black')
plt.text(12.5, 37.5, "Popular", horizontalalignment='center', size=25, color='darkblue')
plt.text(37.5, 37.5, "Polarizing", horizontalalignment='center', size=25, color='darkblue')
plt.text(12.5, 12.5, "Moderate", horizontalalignment='center', size=25, color='darkblue')
plt.text(37.5, 12.5, "Unpopular", horizontalalignment='center', size=25, color='darkblue')
plt.show()

# Vote distribution graph

vote_distribution = pd.read_excel(r'C:\Users\calvi\Data\University\Thesis\Data\Vote_distribution.xlsx')

g = sns.FacetGrid(vote_distribution, col='Horizontal', row='Vertical')
g = g.map_dataframe(sns.lineplot,
                    'X',
                    'Y')
g.set_axis_labels("Ranks", "Votes")
g.set_titles(["Candidate_type", "b", "c", "d"])
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Vote Distribution by Candidate Type')
g.tight_layout()
plt.show()

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

pop_x = np.linspace(1, 8, 300)
bspline = interpolate.make_interp_spline(x, np.array([8, 3.5, 2.5, 2, 1.75, 1.6, 1.6, 1.6]))
pop_y = bspline(pop_x)
plt.plot(pop_x, pop_y)
plt.title("Popular Candidates")
plt.xlabel("Ranks")
plt.ylabel("Votes")
plt.xlim(0, 8)
plt.ylim(0, 9)
plt.xticks([])
plt.yticks([])
plt.show()

pop_x = np.linspace(1, 8, 300)
bspline = interpolate.make_interp_spline(x, np.array([1.6, 1.6, 1.6, 1.75, 2, 2.5, 3.5, 5]))
pop_y = bspline(pop_x)
plt.plot(pop_x, pop_y)
plt.title("Unpopular Candidates")
plt.xlabel("Ranks")
plt.ylabel("Votes")
plt.xlim(0, 8)
plt.ylim(0, 5)
plt.xticks([])
plt.yticks([])
plt.show()

pop_x = np.linspace(1, 8, 300)
bspline = interpolate.make_interp_spline(x, np.array([4, 2, 1.25, 1, 1, 1.25, 2, 4]))
pop_y = bspline(pop_x)
plt.plot(pop_x, pop_y)
plt.title("Polarizing Candidates")
plt.xlabel("Ranks")
plt.ylabel("Votes")
plt.xlim(0, 9)
plt.ylim(0, 5)
plt.xticks([])
plt.yticks([])
plt.show()

pop_x = np.linspace(1, 8, 300)
bspline = interpolate.make_interp_spline(x, np.array([1, 1.75, 3, 4, 4, 3, 1.75, 1]))
pop_y = bspline(pop_x)
plt.plot(pop_x, pop_y)
plt.title("Moderate Candidates")
plt.xlabel("Ranks")
plt.ylabel("Votes")
plt.xlim(0, 9)
plt.ylim(0, 5)
plt.xticks([])
plt.yticks([])
plt.show()


# Original Results


grouped_or_df = df.groupby(['Year', 'To country'])['Points'].sum().reset_index()
original_results = grouped_or_df.groupby(['Year']).apply(lambda x: x.sort_values(['Points'],
                                                         ascending=False)).reset_index(drop=True)
or_5 = original_results.groupby('Year').apply(lambda x: x.nlargest(5, 'Points')).reset_index(drop=True)
or_26 = original_results.groupby('Year').apply(lambda x: x.nlargest(30, 'Points')).reset_index(drop=True)
or_26.to_csv("or.csv")

g = sns.FacetGrid(or_5, col='Year', sharex=False)
g = g.map_dataframe(sns.barplot, 'To country', 'Points', ci=None, palette=sns.color_palette('Reds_r'))
g.set_axis_labels("Country", "Vote Count")
g.set_titles(col_template="{col_name}")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Vote Results by Year')
g.tight_layout()
g.fig.set_figwidth(26.0)
g.fig.set_figheight(5.0)
plt.show()

br_2016 = bootstrap_replicates(df_2016, or_winner, replicates=1000)
br_2016.value_counts(normalize=True)
br_2016.value_counts(normalize=True)

br_2017 = bootstrap_replicates(df_2017, or_winner, replicates=1000)
br_2017.value_counts()
br_2017.value_counts(normalize=True)

br_2018 = bootstrap_replicates(df_2018, or_winner, replicates=1000)
br_2018.value_counts()
br_2018.value_counts(normalize=True)

br_2019 = bootstrap_replicates(df_2019, or_winner, replicates=1000)
br_2019.value_counts()
br_2019.value_counts(normalize=True)

br_2021 = bootstrap_replicates(df_2021, or_winner, replicates=1000)
br_2021.value_counts()
br_2021.value_counts(normalize=True)

br_2016['Year'] = 2016
br_2017['Year'] = 2017
br_2018['Year'] = 2018
br_2019['Year'] = 2019
br_2021['Year'] = 2021

br_all = pd.concat([br_2016, br_2017, br_2018, br_2019, br_2021], ignore_index=True)
br_all['Country'] = br_all[0]

b = sns.FacetGrid(br_all, row='Year', sharex=False)
b = b.map_dataframe(sns.histplot, 'Country', palette=sns.color_palette('Reds_r'))
b.set_axis_labels("Country", "Vote Count")
b.set_titles(row_template="{row_name}")
b.fig.suptitle('Bootstrap Replication Results')
plt.show()

# Plurality Vote Results
grouped_pv_df = df.groupby(['Year', 'To country'])['PV'].sum().reset_index()
pv_results = grouped_pv_df.groupby(['Year']).apply(lambda x: x.sort_values(['PV'],
                                                                           ascending=False)).reset_index(drop=True)

pv_results_5 = pv_results.groupby('Year').apply(lambda x: x.nlargest(5, 'PV')).reset_index(drop=True)
pv_26 = pv_results.groupby('Year').apply(lambda x: x.nlargest(30, 'PV')).reset_index(drop=True)
pv_26.to_csv("pv.csv")

g = sns.FacetGrid(pv_results_5, col='Year', sharex=False)
g = g.map_dataframe(sns.barplot, 'To country', 'PV', ci=None, palette=sns.color_palette('Blues_r'))
g.set_axis_labels("Country", "PV Count")
g.set_titles(col_template="{col_name}")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Plurality Vote Results by Year')
g.tight_layout()
g.fig.set_figwidth(26.0)
g.fig.set_figheight(5.0)
plt.show()

br_2016 = bootstrap_replicates(df_2016, pv_winner, replicates=1000)
br_2016.value_counts(normalize=True)
br_2016.value_counts(normalize=True)

br_2017 = bootstrap_replicates(df_2017, pv_winner, replicates=1000)
br_2017.value_counts()
br_2017.value_counts(normalize=True)

br_2018 = bootstrap_replicates(df_2018, pv_winner, replicates=1000)
br_2018.value_counts()
br_2018.value_counts(normalize=True)

br_2019 = bootstrap_replicates(df_2019, pv_winner, replicates=1000)
br_2019.value_counts()
br_2019.value_counts(normalize=True)

br_2021 = bootstrap_replicates(df_2021, pv_winner, replicates=1000)
br_2021.value_counts()
br_2021.value_counts(normalize=True)

br_2016['Year'] = 2016
br_2017['Year'] = 2017
br_2018['Year'] = 2018
br_2019['Year'] = 2019
br_2021['Year'] = 2021

br_all = pd.concat([br_2016, br_2017, br_2018, br_2019, br_2021], ignore_index=True)
br_all['Country'] = br_all[0]


b = sns.FacetGrid(br_all, row='Year', sharex=False)
b = b.map_dataframe(sns.histplot, 'Country', palette=sns.color_palette('Reds_r'))
b.set_axis_labels("Country", "Vote Count")
b.set_titles(row_template="{row_name}")
b.fig.suptitle('Bootstrap Replication Results')
plt.show()

# Borda Count
grouped_bc_df = df.groupby(['Year', 'To country'])['BC'].sum().reset_index()
bc_results = grouped_bc_df.groupby(['Year']).apply(lambda x: x.sort_values(['BC'],
                                                   ascending=False)).reset_index(drop=True)

bc_results_5 = bc_results.groupby('Year').apply(lambda x: x.nlargest(5, 'BC')).reset_index(drop=True)
bc_26 = bc_results.groupby('Year').apply(lambda x: x.nlargest(30, 'BC')).reset_index(drop=True)
bc_26.to_csv("bc.csv")


g = sns.FacetGrid(bc_results_5, col='Year', sharex=False)
g = g.map_dataframe(sns.barplot, 'To country', 'BC', ci=None, palette=sns.color_palette('Greens_r'))
g.set_axis_labels("Country", "Borda Count")
g.set_titles(col_template="{col_name}")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Borda Count Results by Year')
g.tight_layout()
g.fig.set_figwidth(26.0)
g.fig.set_figheight(5.0)
plt.show()

br_2016 = bootstrap_replicates(df_2016, bc_winner, replicates=1000)
br_2016.value_counts(normalize=True)
br_2016.value_counts(normalize=True)

br_2017 = bootstrap_replicates(df_2017, bc_winner, replicates=1000)
br_2017.value_counts()
br_2017.value_counts(normalize=True)

br_2018 = bootstrap_replicates(df_2018, bc_winner, replicates=1000)
br_2018.value_counts()
br_2018.value_counts(normalize=True)

br_2019 = bootstrap_replicates(df_2019, bc_winner, replicates=1000)
br_2019.value_counts()
br_2019.value_counts(normalize=True)

br_2021 = bootstrap_replicates(df_2021, bc_winner, replicates=1000)
br_2021.value_counts()
br_2021.value_counts(normalize=True)

br_2016['Year'] = 2016
br_2017['Year'] = 2017
br_2018['Year'] = 2018
br_2019['Year'] = 2019
br_2021['Year'] = 2021

br_all = pd.concat([br_2016, br_2017, br_2018, br_2019, br_2021], ignore_index=True)
br_all['Country'] = br_all[0]

b = sns.FacetGrid(br_all, row='Year', sharex=False)
b = b.map_dataframe(sns.histplot, 'Country', palette=sns.color_palette('Blacks_r'))
b.set_axis_labels("Country", "Vote Count")
b.set_titles(row_template="{row_name}")
b.fig.suptitle('Bootstrap Replication Results')
plt.show()

# Alternative Vote

data = df_2021
grouped_av_df = data.groupby(['Year', 'To country'])['AV'].agg({('zero', lambda x: (x == 0).sum()),
                                                               ('one', lambda x: (x == 1).sum()),
                                                               ('two', lambda x: (x == 2).sum()),
                                                               ('three', lambda x: (x == 3).sum()),
                                                               ('four', lambda x: (x == 4).sum()),
                                                               ('five', lambda x: (x == 5).sum()),
                                                               ('six', lambda x: (x == 6).sum()),
                                                               ('seven', lambda x: (x == 7).sum()),
                                                               ('eight', lambda x: (x == 8).sum()),
                                                               ('nine', lambda x: (x == 9).sum()),
                                                               ('ten', lambda x: (x == 10).sum())}).reset_index()


while sum(grouped_av_df['ten']/sum(grouped_av_df['ten']) > 0.5) == 0:
    # Remove countries that did not receive a single top vote
    del_list = grouped_av_df.index[grouped_av_df['ten'] == 0].tolist()
    grouped_av_df = grouped_av_df.drop(labels=del_list, axis=0)
    # Find countries with the fewest top votes
    move_countries = grouped_av_df[grouped_av_df['ten'] == min(grouped_av_df['ten'])]['To country'].unique().tolist()
    move_list = data.index[data['To country'].isin(move_countries) & (data['AV'] == 10)].tolist()
    # Remove their top votes
    data.loc[move_list, 'AV'] = 0
    # Find the voter
    voters = data.loc[move_list]['Voter_ID'].tolist()
    # Change their second vote to the highest vote
    X = data[data['Voter_ID'].isin(voters)]
    change_list = X.index[X['AV'] == X['AV'].max()].tolist()
    data.loc[change_list, 'AV'] = 10
    # Recount
    grouped_av_df = data.groupby(['Year', 'To country'])['AV'].agg({('zero', lambda x: (x == 0).sum()),
                                                                    ('one', lambda x: (x == 1).sum()),
                                                                    ('two', lambda x: (x == 2).sum()),
                                                                    ('three', lambda x: (x == 3).sum()),
                                                                    ('four', lambda x: (x == 4).sum()),
                                                                    ('five', lambda x: (x == 5).sum()),
                                                                    ('six', lambda x: (x == 6).sum()),
                                                                    ('seven', lambda x: (x == 7).sum()),
                                                                    ('eight', lambda x: (x == 8).sum()),
                                                                    ('nine', lambda x: (x == 9).sum()),
                                                                    ('ten', lambda x: (x == 10).sum())}).reset_index()

else:
    print(grouped_av_df.groupby('Year').apply(lambda x: x.nlargest(30, 'ten')).reset_index(drop=True))


br_2016 = bootstrap_replicates(df_2016, av_winner, replicates=1000)
br_2016.value_counts(normalize=True)
br_2016.value_counts(normalize=True)

br_2017 = bootstrap_replicates(df_2017, av_winner, replicates=1000)
br_2017.value_counts()
br_2017.value_counts(normalize=True)

br_2018 = bootstrap_replicates(df_2018, av_winner, replicates=1000)
br_2018.value_counts()
br_2018.value_counts(normalize=True)

br_2019 = bootstrap_replicates(df_2019, av_winner, replicates=1000)
br_2019.value_counts()
br_2019.value_counts(normalize=True)

br_2021 = bootstrap_replicates(df_2021, av_winner, replicates=1000)
br_2021.value_counts()
br_2021.value_counts(normalize=True)


# Approval Voting

grouped_apv_df = df.groupby(['Year', 'To country'])['ApV'].sum().reset_index()
apv_results = grouped_apv_df.groupby(['Year']).apply(lambda x: x.sort_values(['ApV'], ascending=False)).\
                reset_index(drop=True)

apv_results_5 = apv_results.groupby('Year').apply(lambda x: x.nlargest(5, 'ApV')).reset_index(drop=True)
apv_26 = apv_results.groupby('Year').apply(lambda x: x.nlargest(30, 'ApV')).reset_index(drop=True)
apv_26.to_csv("apv.csv")


g = sns.FacetGrid(apv_results_5, col='Year', sharex=False)
g = g.map_dataframe(sns.barplot, 'To country', 'ApV', ci=None, palette=sns.color_palette('Blues_r'))
g.set_axis_labels("Country", "Approval Vote Count")
g.set_titles(col_template="{col_name}")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Approval Vote Results by Year')
g.tight_layout()
g.fig.set_figwidth(26.0)
g.fig.set_figheight(5.0)
plt.show()

br_2016 = bootstrap_replicates(df_2016, apv_winner, replicates=1000)
br_2016.value_counts(normalize=True)
br_2016.value_counts(normalize=True)

br_2017 = bootstrap_replicates(df_2017, apv_winner, replicates=1000)
br_2017.value_counts()
br_2017.value_counts(normalize=True)

br_2018 = bootstrap_replicates(df_2018, apv_winner, replicates=1000)
br_2018.value_counts()
br_2018.value_counts(normalize=True)

br_2019 = bootstrap_replicates(df_2019, apv_winner, replicates=1000)
br_2019.value_counts()
br_2019.value_counts(normalize=True)

br_2021 = bootstrap_replicates(df_2021, apv_winner, replicates=1000)
br_2021.value_counts()
br_2021.value_counts(normalize=True)

br_2016['Year'] = 2016
br_2017['Year'] = 2017
br_2018['Year'] = 2018
br_2019['Year'] = 2019
br_2021['Year'] = 2021

br_all = pd.concat([br_2016, br_2017, br_2018, br_2019, br_2021], ignore_index=True)
br_all['Country'] = br_all[0]

b = sns.FacetGrid(br_all, row='Year', sharex=False)
b = b.map_dataframe(sns.histplot, 'Country', palette=sns.color_palette('Reds_r'))
b.set_axis_labels("Country", "Vote Count")
b.set_titles(row_template="{row_name}")
b.fig.suptitle('Bootstrap Replication Results')
plt.show()

# Coombs' method

cm_winner(df_2021)

br_2016 = bootstrap_replicates(df_2016, cm_winner, replicates=1000)
br_2016.value_counts()
br_2016.value_counts(normalize=True)

br_2017 = bootstrap_replicates(df_2017, cm_winner, replicates=1000, random_state=2)
br_2017.value_counts()
br_2017.value_counts(normalize=True)

br_2018 = bootstrap_replicates(df_2018, cm_winner, replicates=1000)
br_2018.value_counts()
br_2018.value_counts(normalize=True)

br_2019 = \
    bootstrap_replicates(df_2019, cm_winner, replicates=1000)
br_2019.value_counts()
br_2019.value_counts(normalize=True)

br_2021 = \
    bootstrap_replicates(df_2021, cm_winner, replicates=1000, random_state=10)
br_2021.value_counts()
br_2021.value_counts(normalize=True)

br_2016['Year'] = 2016
br_2017['Year'] = 2017
br_2018['Year'] = 2018
br_2019['Year'] = 2019
br_2021['Year'] = 2021

br_all = pd.concat([br_2016, br_2017, br_2018, br_2019, br_2021], ignore_index=True)
br_all['Country'] = br_all[0]

b = sns.FacetGrid(br_all, row='Year', sharex=False)
b = b.map_dataframe(sns.histplot, 'Country', palette=sns.color_palette('Reds_r'))
b.set_axis_labels("Country", "Vote Count")
b.set_titles(row_template="{row_name}")
b.fig.suptitle('Bootstrap Replication Results')
plt.show()

# Rank-order difference test

or_2016 = or_26[or_26['Year'] == 2016]
or_2017 = or_26[or_26['Year'] == 2017]
or_2018 = or_26[or_26['Year'] == 2018]
or_2019 = or_26[or_26['Year'] == 2019]
or_2021 = or_26[or_26['Year'] == 2021]

pv_2016 = pv_26[pv_26['Year'] == 2016]
pv_2017 = pv_26[pv_26['Year'] == 2017]
pv_2018 = pv_26[pv_26['Year'] == 2018]
pv_2019 = pv_26[pv_26['Year'] == 2019]
pv_2021 = pv_26[pv_26['Year'] == 2021]

bc_2016 = bc_26[bc_26['Year'] == 2016]
bc_2017 = bc_26[bc_26['Year'] == 2017]
bc_2018 = bc_26[bc_26['Year'] == 2018]
bc_2019 = bc_26[bc_26['Year'] == 2019]
bc_2021 = bc_26[bc_26['Year'] == 2021]

apv_2016 = apv_26[apv_26['Year'] == 2016]
apv_2017 = apv_26[apv_26['Year'] == 2017]
apv_2018 = apv_26[apv_26['Year'] == 2018]
apv_2019 = apv_26[apv_26['Year'] == 2019]
apv_2021 = apv_26[apv_26['Year'] == 2021]


def rocc(method1, method2, func='kendalltau'):
    """Determines the correlation coefficient between two rank orders"""
    coefficients = []
    if func == 'spearmanr':
        for i in range(27):
            coeff, p_value = stats.spearmanr(method1['To country'].head(i), method2['To country'].head(i))
            coefficients.append(coeff)
    elif func == 'kendalltau':
        for i in range(27):
            coeff, p_value = stats.kendalltau(method1['To country'].head(i), method2['To country'].head(i))
            coefficients.append(coeff)
    else:
        coefficients = 'ERROR'
    return coefficients


rocc(or_2021, apv_2021, func='kendalltau')

# Rank-order graph

rank_order_df = pd.read_excel(r'C:\Users\calvi\Data\University\Thesis\Data\Rank_Order_Data.xlsx')

# Spearman

fig = plt.figure()
fig.set_figheight(7)
fig.set_figwidth(9)
r = sns.lineplot(data=rank_order_df, x='Ranks', y='Spearman_coeff', hue='Method')
r.set(xlabel="Number of candidates included in rank-order",
      ylabel="Spearman's Rank-order Correlation Coefficient",
      title="Voting Method Difference Test (Spearman's coeff)")
plt.axhline(0, 0, 25, color='Black', linewidth=0.5)
plt.axhline(0.5, 0, 25, color='Grey', linestyle='--', linewidth=0.5)
plt.axhline(-0.5, 0, 25, color='Grey', linestyle='--', linewidth=0.5)
plt.ylim(-1, 1)
plt.xlim(0, 27)
plt.show()


s = sns.FacetGrid(rank_order_df, col='Year')
s = s.map_dataframe(sns.lineplot, 'Ranks', 'Spearman_coeff', hue='Method', ci=None)
s.set_axis_labels("Number of candidates included in rank-order", "Spearman's Rank-order Correlation Coefficient")
s.set_titles(col_template="{col_name}")
s.fig.subplots_adjust(top=0.9)
s.fig.suptitle("Voting Method Difference Test by year (Spearman's coeff)")
s.tight_layout()
s.fig.set_figwidth(18.0)
s.fig.set_figheight(4.0)
s.add_legend()
plt.show()

# Kendall's tau

fig = plt.figure()
fig.set_figheight(7)
fig.set_figwidth(9)
r = sns.lineplot(data=rank_order_df, x='Ranks', y='Kendalls tau', hue='Method')
r.set(xlabel="Number of candidates included in rank-order",
      ylabel="Kendalls's tau",
      title="Voting Method Difference Test (Kendall's tau)")
plt.axhline(0, 0, 25, color='Black', linewidth=0.5)
plt.axhline(0.5, 0, 25, color='Grey', linestyle='--', linewidth=0.5)
plt.axhline(-0.5, 0, 25, color='Grey', linestyle='--', linewidth=0.5)
plt.ylim(-1, 1)
plt.xlim(0, 27)
plt.show()


s = sns.FacetGrid(rank_order_df, col='Year')
s = s.map_dataframe(sns.lineplot, 'Ranks', 'Kendalls tau', hue='Method', ci=None)
s.set_axis_labels("Number of candidates included in rank-order", "Kendall's tau")
s.set_titles(col_template="{col_name}")
s.fig.subplots_adjust(top=0.9)
s.fig.suptitle("Voting Method Difference Test by year (Kendall's tau)")
s.tight_layout()
s.fig.set_figwidth(18.0)
s.fig.set_figheight(4.0)
s.add_legend()
plt.show()

# Holm-Bonferroni Test
pvals = [1, 2]

statsmodels.stats.multitest.multipletests(pvals, alpha=0.10, method='holm')

# Coombs alternate

Albania = Candidate("Albania")
Armenia = Candidate("Armenia")
Australia = Candidate("Australia")
Austria = Candidate("Austria")
Azerbaijan = Candidate("Azerbaijan")
Belarus = Candidate("Belarus")
Belgium = Candidate("Belgium")
Bulgaria = Candidate("Bulgaria")
Croatia = Candidate("Croatia")
Cyprus = Candidate("Cyprus")
Czech_Republic = Candidate("Czech Republic")
Denmark = Candidate("Denmark")
Estonia = Candidate("Estonia")
Finland = Candidate("Finland")
France = Candidate("France")
Georgia = Candidate("Georgia")
Germany = Candidate("Germany")
Greece = Candidate("Greece")
Hungary = Candidate("Hungary")
Iceland = Candidate("Iceland")
Ireland = Candidate("Ireland")
Israel = Candidate("Israel")
Italy = Candidate("Italy")
Latvia = Candidate("Latvia")
Lithuania = Candidate("Lithuania")
Malta = Candidate("Malta")
Moldova = Candidate("Moldova")
Netherlands = Candidate("Netherlands")
North_Macedonia = Candidate("North Macedonia")
Norway = Candidate("Norway")
Poland = Candidate("Poland")
Portugal = Candidate("Portugal")
Romania = Candidate("Romania")
Russia = Candidate("Russia")
San_Marino = Candidate("San Marino")
Serbia = Candidate("Serbia")
Slovenia = Candidate("Slovenia")
Spain = Candidate("Spain")
Sweden = Candidate("Sweden")
Switzerland = Candidate("Switzerland")
The_Netherlands = Candidate("The Netherlands")
Ukraine = Candidate("Ukraine")
United_Kingdom = Candidate("United Kingdom")

candidates = [Albania, Armenia, Australia, Austria, Azerbaijan, Belarus, Belgium, Bulgaria, Croatia, Cyprus,
              Czech_Republic, Denmark, Estonia, Finland, France, Georgia, Germany, Greece, Hungary, Iceland, Ireland,
              Israel, Italy, Latvia, Lithuania, Malta, Moldova, Netherlands, North_Macedonia, Norway, Poland, Portugal,
              Romania, Russia, San_Marino, Serbia, Slovenia, Spain, Sweden, Switzerland, The_Netherlands, Ukraine,
              United_Kingdom]

ballots_2016 = \
    [Ballot(ranked_candidates=[Australia, France, Italy, Russia, Spain, United_Kingdom, Bulgaria, Israel,
                                          Malta, Hungary]),
                Ballot(ranked_candidates=[Australia, Italy, Bulgaria, Russia, Ukraine, Poland, Lithuania, Sweden,
                                          Armenia, Hungary]),
                Ballot(ranked_candidates=[France, Georgia, Malta, Bulgaria, Cyprus, Australia, Belgium, Spain, Russia,
                                          Czech_Republic]),
                Ballot(ranked_candidates=[Russia, Ukraine, Georgia, France, Cyprus, Malta, Austria, Hungary, Sweden,
                                          Poland]),
                Ballot(ranked_candidates=[Belgium, Israel, Bulgaria, Lithuania, France, Spain, United_Kingdom,
                                          The_Netherlands, Ukraine, Croatia]),
                Ballot(ranked_candidates=[Belgium, Bulgaria, Ukraine, France, Malta, Russia, United_Kingdom, Austria,
                                          Spain, Lithuania]),
                Ballot(ranked_candidates=[Australia, Malta, Sweden, France, Croatia, Belgium, Czech_Republic, Russia,
                                          Armenia, Lithuania]),
                Ballot(ranked_candidates=[Poland, Ukraine, Russia, Sweden, The_Netherlands, Bulgaria, Serbia, Australia,
                                          Germany, France]),
                Ballot(ranked_candidates=[Russia, Ukraine, Bulgaria, Australia, Malta, Lithuania, Hungary, Israel,
                                          Poland, Spain]),
                Ballot(ranked_candidates=[Russia, Ukraine, Bulgaria, Hungary, Israel, Malta, France, Poland, Australia,
                                          Georgia]),
                Ballot(ranked_candidates=[Russia, Lithuania, Sweden, Ukraine, Australia, Malta, Belgium, Armenia,
                                          Azerbaijan, Bulgaria]),
                Ballot(ranked_candidates=[Russia, Ukraine, Azerbaijan, Armenia, Poland, Latvia, Bulgaria, Hungary,
                                          Sweden, Australia]),
                Ballot(ranked_candidates=[Australia, Italy, France, Georgia, Bulgaria, Austria, The_Netherlands,
                                          Ukraine, Israel, Czech_Republic]),
                Ballot(ranked_candidates=[Poland, The_Netherlands, France, Armenia, Russia, Bulgaria, Australia,
                                          Austria, Ukraine, Sweden]),
                Ballot(ranked_candidates=[Ukraine, Australia, Serbia, France, Czech_Republic, Russia, Croatia,
                                          Bulgaria, Armenia, Azerbaijan]),
                Ballot(ranked_candidates=[Serbia, Croatia, Azerbaijan, Ukraine, Russia, Austria, France, Australia,
                                          Bulgaria, Hungary]),
                Ballot(ranked_candidates=[Armenia, Belgium, Australia, Malta, Russia, Serbia, Spain, Georgia,
                                          The_Netherlands, Lithuania]),
                Ballot(ranked_candidates=[Russia, Ukraine, Armenia, Cyprus, Poland, Australia, Austria, Sweden, Israel,
                                          Azerbaijan]),
                Ballot(ranked_candidates=[Australia, Czech_Republic, France, Israel, Russia, Hungary, Armenia, Italy,
                                          Serbia, Cyprus]),
                Ballot(ranked_candidates=[Serbia, Ukraine, Russia, Sweden, Austria, Australia, Poland, Hungary, France,
                                          Bulgaria]),
                Ballot(ranked_candidates=[Russia, Australia, Armenia, France, Malta, Israel, Hungary, Italy, Azerbaijan,
                                          Croatia]),
                Ballot(ranked_candidates=[Bulgaria, Russia, Armenia, Ukraine, France, Australia, Hungary, Lithuania,
                                          Poland, Sweden]),
                Ballot(ranked_candidates=[Sweden, Hungary, Croatia, The_Netherlands, Malta, France, United_Kingdom,
                                          Lithuania, Belgium, Latvia]),
                Ballot(ranked_candidates=[Ukraine, Russia, Armenia, Poland, Azerbaijan, Bulgaria, Austria, Hungary,
                                          Sweden, Australia]),
                Ballot(ranked_candidates=[Ukraine, Australia, Belgium, The_Netherlands, Bulgaria, Lithuania, Sweden,
                                          United_Kingdom, Israel, Spain]),
                Ballot(ranked_candidates=[Sweden, Australia, Belgium, The_Netherlands, Lithuania, Poland, Russia,
                                          Ukraine, France, Austria]),
                Ballot(ranked_candidates=[Sweden, Australia, Latvia, Ukraine, The_Netherlands, Lithuania, Cyprus,
                                          United_Kingdom, Malta, France]),
                Ballot(ranked_candidates=[Russia, Sweden, Ukraine, Latvia, Austria, Lithuania, Australia, Cyprus,
                                          France, Poland]),
                Ballot(ranked_candidates=[Ukraine, Bulgaria, Australia, Serbia, Croatia, Israel, Belgium, Latvia,
                                          The_Netherlands, Azerbaijan]),
                Ballot(ranked_candidates=[Serbia, Bulgaria, Russia, Armenia, Ukraine, Croatia, Belgium, Australia,
                                          Poland, Italy]),
                Ballot(ranked_candidates=[Sweden, The_Netherlands, Australia, Israel, Lithuania, Malta, Austria,
                                          Czech_Republic, Italy, France]),
                Ballot(ranked_candidates=[Ukraine, Sweden, Russia, Australia, Austria, Poland, Bulgaria, France,
                                          Latvia, Cyprus]),
                Ballot(ranked_candidates=[Italy, Bulgaria, Austria, The_Netherlands, Australia, Sweden, Malta, Spain,
                                          Armenia, Russia]),
                Ballot(ranked_candidates=[Armenia, Ukraine, Austria, Poland, Russia, Bulgaria, Belgium, Israel,
                                          Sweden, Spain]),
                Ballot(ranked_candidates=[Ukraine, Belgium, Australia, Latvia, Sweden, Lithuania, France, Israel,
                                          Croatia, Germany]),
                Ballot(ranked_candidates=[Armenia, Ukraine, Russia, Azerbaijan, Latvia, Lithuania, Poland, Bulgaria,
                                          Hungary, Australia]),
                Ballot(ranked_candidates=[Israel, Sweden, Georgia, Ukraine, Australia, Belgium, The_Netherlands, Italy,
                                          Latvia, Lithuania]),
                Ballot(ranked_candidates=[Russia, Poland, Sweden, Austria, Ukraine, Australia, Bulgaria,
                                          The_Netherlands, Armenia, Italy]),
                Ballot(ranked_candidates=[Russia, Armenia, Cyprus, Australia, France, Georgia, Malta, Hungary, Ukraine,
                                          Azerbaijan]),
                Ballot(ranked_candidates=[Cyprus, Russia, Armenia, Bulgaria, Ukraine, Australia, France, Poland,
                                          Hungary, Austria]),
                Ballot(ranked_candidates=[Australia, Malta, Latvia, Azerbaijan, The_Netherlands, Spain, Sweden,
                                          Lithuania, Czech_Republic, Armenia]),
                Ballot(ranked_candidates=[Ukraine, Russia, Poland, Sweden, Austria, Cyprus, Bulgaria, Australia,
                                          The_Netherlands, France]),
                Ballot(ranked_candidates=[The_Netherlands, Australia, Russia, Croatia, Sweden, Czech_Republic, Malta,
                                          Belgium, France, Austria]),
                Ballot(ranked_candidates=[Sweden, Poland, Australia, Russia, The_Netherlands, France, Lithuania,
                                          Belgium, Austria, Azerbaijan]),
                Ballot(ranked_candidates=[Belgium, Bulgaria, The_Netherlands, United_Kingdom, Italy, Sweden, Israel,
                                          France, Czech_Republic, Latvia]),
                Ballot(ranked_candidates=[Lithuania, Poland, Russia, Latvia, Australia, Bulgaria, Ukraine,
                                          United_Kingdom, Sweden, Austria]),
                Ballot(ranked_candidates=[Ukraine, Australia, France, Bulgaria, Belgium, Latvia, Spain, Croatia,
                                          Armenia, Lithuania]),
                Ballot(ranked_candidates=[France, Russia, Ukraine, Bulgaria, Armenia, Australia, Poland, Austria,
                                          Azerbaijan, Belgium]),
                Ballot(ranked_candidates=[Spain, Ukraine, Israel, France, Australia, Malta, The_Netherlands, Georgia,
                                          Sweden, Cyprus]),
                Ballot(ranked_candidates=[Ukraine, Poland, Russia, Bulgaria, Cyprus, France, Serbia, Georgia, Lithuania,
                                          Armenia]),
                Ballot(ranked_candidates=[Ukraine, Lithuania, Sweden, Russia, Armenia, Australia, Malta,
                                          The_Netherlands, Israel, Bulgaria]),
                Ballot(ranked_candidates=[Russia, Ukraine, Lithuania, Sweden, Australia, Poland, Austria, Azerbaijan,
                                          Georgia, Bulgaria]),
                Ballot(ranked_candidates=[Australia, Georgia, Ukraine, Latvia, Israel, Belgium, Armenia, Poland,
                                          Bulgaria, France]),
                Ballot(ranked_candidates=[Latvia, Ukraine, Russia, Sweden, Poland, Australia, Georgia, France, Hungary,
                                          Austria]),
                Ballot(ranked_candidates=[United_Kingdom, Bulgaria, Italy, Armenia, France, Cyprus, Russia, Australia,
                                          Spain, Israel]),
                Ballot(ranked_candidates=[Australia, Russia, Bulgaria, Italy, Azerbaijan, Hungary, Ukraine,
                                          The_Netherlands, France, United_Kingdom]),
                Ballot(ranked_candidates=[Ukraine, Australia, Spain, Russia, Sweden, Armenia, Lithuania, Malta, Cyprus,
                                          Croatia]),
                Ballot(ranked_candidates=[Russia, Ukraine, Azerbaijan, Armenia, France, Australia, Austria, Lithuania,
                                          Bulgaria, Latvia]),
                Ballot(ranked_candidates=[Malta, Armenia, Russia, Hungary, Serbia, France, Australia, Belgium, Bulgaria,
                                          Poland]),
                Ballot(ranked_candidates=[Serbia, Russia, Ukraine, Azerbaijan, Croatia, Bulgaria, Italy, Hungary,
                                          Lithuania, France]),
                Ballot(ranked_candidates=[Italy, Australia, Bulgaria, Spain, The_Netherlands, Belgium, Ukraine, Israel,
                                          Lithuania, Poland]),
                Ballot(ranked_candidates=[Lithuania, Poland, Australia, Sweden, Russia, Bulgaria, Ukraine, Latvia,
                                          Belgium, France]),
                Ballot(ranked_candidates=[Ukraine, Australia, Georgia, Israel, Latvia, Spain, Czech_Republic, Bulgaria,
                                          Lithuania, France]),
                Ballot(ranked_candidates=[Ukraine, Sweden, Russia, Australia, Hungary, Latvia, Austria, Bulgaria,
                                          Armenia, Cyprus]),
                Ballot(ranked_candidates=[Armenia, Azerbaijan, Malta, France, United_Kingdom, Georgia, Cyprus, Latvia,
                                          Australia, Hungary]),
                Ballot(ranked_candidates=[Armenia, Ukraine, Austria, Cyprus, Azerbaijan, Poland, Australia, France,
                                          Sweden, Latvia]),
                Ballot(ranked_candidates=[Ukraine, Italy, United_Kingdom, Russia, Georgia, Cyprus, The_Netherlands,
                                          Malta, Hungary, Austria]),
                Ballot(ranked_candidates=[Ukraine, Russia, Lithuania, Poland, Latvia, Australia, Sweden, Bulgaria,
                                          Armenia, Hungary]),
                Ballot(ranked_candidates=[Ukraine, Malta, Lithuania, Israel, Australia, The_Netherlands, Bulgaria,
                                          Armenia, United_Kingdom, Russia]),
                Ballot(ranked_candidates=[Russia, Hungary, Bulgaria, Ukraine, Australia, Belgium, Croatia, Cyprus,
                                          Armenia, Sweden]),
                Ballot(ranked_candidates=[Ukraine, Bulgaria, Belgium, Latvia, Australia, Serbia, Armenia, Hungary,
                                          Israel, The_Netherlands]),
                Ballot(ranked_candidates=[Serbia, Russia, Croatia, Ukraine, Austria, Sweden, Poland, Australia,
                                          Bulgaria, Italy]),
                Ballot(ranked_candidates=[Armenia, Hungary, Australia, France, Malta, Italy, Russia, The_Netherlands,
                                          Azerbaijan, Israel]),
                Ballot(ranked_candidates=[Bulgaria, France, Russia, Ukraine, Armenia, Poland, Australia, Italy, Austria,
                                          Sweden]),
                Ballot(ranked_candidates=[Australia, Azerbaijan, France, Malta, Russia, The_Netherlands, Armenia,
                                          Czech_Republic, Lithuania, Spain]),
                Ballot(ranked_candidates=[Australia, Poland, Russia, Ukraine, Lithuania, Austria, Bulgaria, France,
                                          The_Netherlands, Spain]),
                Ballot(ranked_candidates=[Australia, Belgium, Israel, Lithuania, Ukraine, The_Netherlands, Austria,
                                          Latvia, Italy, Bulgaria]),
                Ballot(ranked_candidates=[Serbia, Austria, Germany, Italy, Russia, Poland, Ukraine, France, Spain,
                                          Australia]),
                Ballot(ranked_candidates=[Australia, Sweden, Austria, Israel, Italy, France, Belgium, Ukraine, Armenia,
                                          Azerbaijan]),
                Ballot(ranked_candidates=[Belgium, Poland, Armenia, Ukraine, Austria, Australia, France, Russia, Sweden,
                                          Bulgaria]),
                Ballot(ranked_candidates=[Lithuania, Belgium, Latvia, Azerbaijan, Israel, Australia, Sweden, Georgia,
                                          Hungary, Bulgaria]),
                Ballot(ranked_candidates=[Russia, Azerbaijan, Poland, Armenia, Georgia, Latvia, Australia, Lithuania,
                                          Bulgaria, Sweden]),
                Ballot(ranked_candidates=[Georgia, Ukraine, Australia, Cyprus, France, Bulgaria, Lithuania, Israel,
                                          Serbia, Croatia]),
                Ballot(ranked_candidates=[Lithuania, Poland, Bulgaria, Russia, Australia, Ukraine, Spain, Latvia,
                                          Cyprus, Sweden])
                ]

ballots_2017 = \
    [Ballot(ranked_candidates=[Italy, Bulgaria, United_Kingdom, Armenia, Portugal, France, Australia, Romania,
                               Azerbaijan, Greece]),
     Ballot(ranked_candidates=[Italy, Bulgaria, Portugal, Croatia, Hungary, Belgium, Romania, Greece, Sweden, France]),
     Ballot(ranked_candidates=[Portugal, Greece, Moldova, Cyprus, Sweden, Norway, United_Kingdom, France, Italy,
                               Austria]),
     Ballot(ranked_candidates=[Cyprus, Portugal, France, Bulgaria, Moldova, Belgium, Italy, Sweden, Romania, Greece]),
     Ballot(ranked_candidates=[United_Kingdom, Moldova, Bulgaria, Portugal, Sweden, Denmark, The_Netherlands, Austria,
                               Poland, Armenia]),
     Ballot(ranked_candidates=[Moldova, Romania, Denmark, Portugal, Sweden, Bulgaria, Belgium, United_Kingdom, Italy,
                               Croatia]),
     Ballot(ranked_candidates=[The_Netherlands, Bulgaria, Portugal, Belarus, Italy, Hungary, Sweden, Norway, Belgium,
                               United_Kingdom]),
     Ballot(ranked_candidates=[Portugal, Romania, Belgium, Bulgaria, Italy, Hungary, Croatia, Moldova, Poland, Sweden]),
     Ballot(ranked_candidates=[Belarus, Moldova, Portugal, Ukraine, Poland, Greece, Israel, Romania, Bulgaria,
                               Croatia]),
     Ballot(ranked_candidates=[Bulgaria, Moldova, Portugal, Hungary, Italy, France, Belgium, Sweden, Romania, Israel]),
     Ballot(ranked_candidates=[Bulgaria, Portugal, Sweden, Australia, Norway, The_Netherlands, Austria, Denmark, Poland,
                               Israel]),
     Ballot(ranked_candidates=[Bulgaria, Moldova, Hungary, Portugal, Belgium, Norway, France, Sweden, Romania,
                               Ukraine]),
     Ballot(ranked_candidates=[Sweden, Bulgaria, Portugal, Norway, Australia, Cyprus, Azerbaijan, Austria,
                               United_Kingdom, France]),
     Ballot(ranked_candidates=[Portugal, The_Netherlands, Bulgaria, Moldova, Romania, Italy, Poland, France, Sweden,
                               Hungary]),
     Ballot(ranked_candidates=[Austria, The_Netherlands, Israel, Poland, Greece, Azerbaijan, Armenia, Romania, Belarus,
                               Ukraine]),
     Ballot(ranked_candidates=[France, Moldova, Romania, Portugal, Hungary, Greece, Belgium, Cyprus, Sweden, Croatia]),
     Ballot(ranked_candidates=[Hungary, Bulgaria, The_Netherlands, Portugal, Australia, Belarus, Sweden, Denmark,
                               Norway, Austria]),
     Ballot(ranked_candidates=[Hungary, Portugal, Bulgaria, Italy, Romania, Belgium, Moldova, France, Sweden,
                               The_Netherlands]),
     Ballot(ranked_candidates=[Greece, Italy, Portugal, Bulgaria, Sweden, United_Kingdom, Australia, Belgium, France,
                               Azerbaijan]),
     Ballot(ranked_candidates=[Greece, Bulgaria, Italy, Portugal, Moldova, Sweden, Romania, France, Belgium, Armenia]),
     Ballot(ranked_candidates=[Portugal, Australia, The_Netherlands, Sweden, Denmark, United_Kingdom, Austria, Moldova,
                               Bulgaria, Poland]),
     Ballot(ranked_candidates=[Bulgaria, Azerbaijan, Portugal, Ukraine, Romania, Moldova, Belgium, Belarus, Armenia,
                               Italy]),
     Ballot(ranked_candidates=[Sweden, Portugal, Australia, Austria, Norway, The_Netherlands, Bulgaria, Moldova,
                               Belgium, Armenia]),
     Ballot(ranked_candidates=[Sweden, Bulgaria, Moldova, Norway, Belgium, Portugal, Romania, Hungary, Australia,
                               The_Netherlands]),
     Ballot(ranked_candidates=[Bulgaria, Norway, Portugal, Australia, United_Kingdom, Denmark, The_Netherlands, Sweden,
                               Belgium, Hungary]),
     Ballot(ranked_candidates=[Belgium, Portugal, Hungary, Bulgaria, Romania, Italy, Moldova, Sweden, Belarus, France]),
     Ballot(ranked_candidates=[Bulgaria, Portugal, Australia, Denmark, Moldova, France, Romania, United_Kingdom, Norway,
                               Sweden]),
     Ballot(ranked_candidates=[Bulgaria, Croatia, Italy, Portugal, France, Sweden, Belgium, Hungary, Moldova, Israel]),
     Ballot(ranked_candidates=[Sweden, Australia, Portugal, Italy, Bulgaria, Cyprus, Denmark, Moldova, United_Kingdom,
                               Norway]),
     Ballot(ranked_candidates=[Portugal, Belgium, Italy, Bulgaria, Sweden, Moldova, Hungary, Romania, Norway, France]),
     Ballot(ranked_candidates=[Portugal, Italy, Sweden, Norway, Israel, Bulgaria, The_Netherlands, United_Kingdom,
                               Australia, Austria]),
     Ballot(ranked_candidates=[Portugal, Romania, Belgium, Moldova, Armenia, Italy, Bulgaria, Israel, Hungary, Poland]),
     Ballot(ranked_candidates=[Portugal, Azerbaijan, Sweden, Norway, Bulgaria, Austria, Cyprus, Armenia, Italy,
                               The_Netherlands]),
     Ballot(ranked_candidates=[Azerbaijan, Armenia, Portugal, Bulgaria, Belarus, Italy, Ukraine, France, Cyprus,
                               Sweden]),
     Ballot(ranked_candidates=[Norway, Portugal, Bulgaria, The_Netherlands, United_Kingdom, Australia, Sweden, Croatia,
                               Austria, Hungary]),
     Ballot(ranked_candidates=[Portugal, Belgium, Croatia, Moldova, Romania, Bulgaria, Italy, Hungary, Poland,
                               The_Netherlands]),
     Ballot(ranked_candidates=[Cyprus, Azerbaijan, Armenia, Moldova, Romania, Portugal, Italy, Belarus, Bulgaria,
                               United_Kingdom]),
     Ballot(ranked_candidates=[Cyprus, Bulgaria, Portugal, Italy, Moldova, Belgium, France, Sweden, Armenia, Romania]),
     Ballot(ranked_candidates=[Portugal, Bulgaria, The_Netherlands, Australia, Norway, Denmark, Azerbaijan,
                               United_Kingdom, Armenia, Sweden]),
     Ballot(ranked_candidates=[Bulgaria, Moldova, Belgium, Portugal, Romania, Croatia, Italy, Sweden, Cyprus, France]),
     Ballot(ranked_candidates=[Portugal, Australia, Sweden, United_Kingdom, Bulgaria, Italy, Hungary, Denmark, Belgium,
                               Norway]),
     Ballot(ranked_candidates=[Portugal, Belgium, Sweden, Italy, Moldova, Romania, Bulgaria, Poland, Hungary, Norway]),
     Ballot(ranked_candidates=[Belgium, Bulgaria, Romania, Austria, Sweden, Portugal, The_Netherlands, Germany, Norway,
                               Azerbaijan]),
     Ballot(ranked_candidates=[Romania, Portugal, Moldova, Poland, Bulgaria, Belgium, United_Kingdom, Croatia, Hungary,
                               Italy]),
     Ballot(ranked_candidates=[Portugal, Sweden, Belgium, Bulgaria, France, Norway, Ukraine, Moldova, Italy, Belarus]),
     Ballot(ranked_candidates=[Portugal, Bulgaria, Italy, Romania, Belgium, Moldova, France, Sweden, Belarus, Hungary]),
     Ballot(ranked_candidates=[Azerbaijan, Sweden, Moldova, Belgium, France, Portugal, Armenia, Norway, Bulgaria,
                               Austria]),
     Ballot(ranked_candidates=[Moldova, Romania, Bulgaria, Ukraine, Croatia, Portugal, Hungary, Poland, Belgium,
                               France]),
     Ballot(ranked_candidates=[Portugal, Belgium, Bulgaria, Norway, Austria, Australia, United_Kingdom, Romania,
                               Belarus, Moldova]),
     Ballot(ranked_candidates=[Belgium, Portugal, Moldova, Bulgaria, Belarus, Romania, Sweden, Italy, Hungary, Norway]),
     Ballot(ranked_candidates=[Portugal, Norway, Italy, Bulgaria, Sweden, Austria, Belgium, Armenia, The_Netherlands,
                               Australia]),
     Ballot(ranked_candidates=[Portugal, Belgium, Moldova, Bulgaria, Norway, Italy, Romania, Sweden, Hungary, Belarus]),
     Ballot(ranked_candidates=[Italy, Portugal, Bulgaria, Israel, Croatia, Romania, France, Hungary, Belarus, Armenia]),
     Ballot(ranked_candidates=[Italy, Bulgaria, Portugal, Sweden, Romania, Belgium, United_Kingdom, Croatia, Hungary,
                               Moldova]),
     Ballot(ranked_candidates=[Romania, Bulgaria, Sweden, Portugal, Armenia, Azerbaijan, Australia, Belarus, Austria,
                               Hungary]),
     Ballot(ranked_candidates=[Romania, Azerbaijan, Bulgaria, Greece, Portugal, France, Italy, Sweden, Belgium,
                               Belarus]),
     Ballot(ranked_candidates=[Greece, Romania, Italy, Sweden, Belgium, Croatia, Armenia, France, Bulgaria, Austria]),
     Ballot(ranked_candidates=[Croatia, Italy, Portugal, Hungary, Bulgaria, Azerbaijan, Belgium, Moldova, France,
                               Cyprus]),
     Ballot(ranked_candidates=[Bulgaria, Portugal, Denmark, Italy, Sweden, Israel, The_Netherlands, Australia, Moldova,
                               United_Kingdom]),
     Ballot(ranked_candidates=[Portugal, Romania, Bulgaria, Belgium, Moldova, Sweden, Hungary, Poland, Italy, Croatia]),
     Ballot(ranked_candidates=[Portugal, Belgium, The_Netherlands, Australia, Bulgaria, United_Kingdom, Sweden, Norway,
                               Armenia, Azerbaijan]),
     Ballot(ranked_candidates=[Belgium, Portugal, Bulgaria, Romania, Moldova, Hungary, Belarus, Sweden, Ukraine,
                               Croatia]),
     Ballot(ranked_candidates=[Azerbaijan, Austria, Belgium, Sweden, Moldova, Australia, France, The_Netherlands,
                               Norway, Cyprus]),
     Ballot(ranked_candidates=[Moldova, Belgium, Bulgaria, Romania, France, Spain, Italy, Ukraine, Hungary, Sweden]),
     Ballot(ranked_candidates=[The_Netherlands, Bulgaria, Moldova, Sweden, Portugal, Denmark, Australia, Croatia, Israel, Austria]),
     Ballot(ranked_candidates=[Moldova, Hungary, Bulgaria, Portugal, Italy, Belgium, France, Croatia, Sweden, Greece]),
     Ballot(ranked_candidates=[Portugal, Norway, Belgium, The_Netherlands, United_Kingdom, Azerbaijan, Armenia, Italy,
                               Bulgaria, Poland]),
     Ballot(ranked_candidates=[Bulgaria, Italy, Moldova, Portugal, Romania, Belgium, Hungary, Croatia, Sweden,
                               Azerbaijan]),
     Ballot(ranked_candidates=[Portugal, Hungary, Italy, Moldova, Bulgaria, Armenia, Austria, Australia, Sweden,
                               Israel]),
     Ballot(ranked_candidates=[Hungary, Bulgaria, Portugal, Moldova, Italy, Croatia, Romania, Belgium, France, Sweden]),
     Ballot(ranked_candidates=[Portugal, United_Kingdom, Bulgaria, Australia, Sweden, Belgium, Poland, Armenia, Italy,
                               The_Netherlands]),
     Ballot(ranked_candidates=[Croatia, Italy, Portugal, Bulgaria, Belgium, Hungary, Azerbaijan, Moldova, Romania,
                               Sweden]),
     Ballot(ranked_candidates=[Portugal, Italy, Australia, Moldova, Bulgaria, Sweden, Belgium, Austria, Greece,
                               The_Netherlands]),
     Ballot(ranked_candidates=[Portugal, Bulgaria, Italy, Romania, Moldova, Sweden, Belgium, France, The_Netherlands,
                               United_Kingdom]),
     Ballot(ranked_candidates=[Portugal, Australia, Moldova, Bulgaria, Italy, Denmark, Austria, The_Netherlands, Cyprus,
                               Belgium]),
     Ballot(ranked_candidates=[Belgium, Portugal, Moldova, Bulgaria, Norway, Poland, Hungary, Romania, Croatia, Italy]),
     Ballot(ranked_candidates=[Portugal, Sweden, Bulgaria, The_Netherlands, Belgium, Norway, Australia, Hungary, Italy,
                               France]),
     Ballot(ranked_candidates=[Portugal, Italy, Croatia, Belgium, Bulgaria, Hungary, Romania, Germany, France, Sweden]),
     Ballot(ranked_candidates=[Portugal, Bulgaria, Denmark, Norway, Sweden, United_Kingdom, Australia, Austria, Belgium,
                               Romania]),
     Ballot(ranked_candidates=[Portugal, Belgium, Bulgaria, Romania, Hungary, Moldova, Sweden, Croatia, Italy, Poland]),
     Ballot(ranked_candidates=[Belarus, Portugal, Hungary, Croatia, Azerbaijan, France, Moldova, The_Netherlands,
                               Australia, Belgium]),
     Ballot(ranked_candidates=[Moldova, Portugal, Belarus, Sweden, France, Belgium, Hungary, Romania, Bulgaria,
                               Norway]),
     Ballot(ranked_candidates=[Portugal, Australia, Sweden, Bulgaria, Moldova, Belgium, The_Netherlands, Austria,
                               Denmark, Belarus]),
     Ballot(ranked_candidates=[Bulgaria, Poland, Portugal, Romania, Moldova, Croatia, Sweden, Belgium, Italy, Hungary])
     ]

ballots_2018 = [Ballot(ranked_candidates=[Italy, Cyprus, Bulgaria, France, Germany, Israel, Sweden, Serbia, Moldova, Czech_Republic]),
Ballot(ranked_candidates=[Italy, Cyprus, Germany, Ireland, Bulgaria, France, Estonia, United_Kingdom, Austria, Israel]),
Ballot(ranked_candidates=[Sweden, Moldova, Israel, Cyprus, Czech_Republic, Germany, France, The_Netherlands, Australia, Austria]),
Ballot(ranked_candidates=[Cyprus, Israel, Czech_Republic, Norway, Austria, France, Ukraine, Italy, Estonia, Moldova]),
Ballot(ranked_candidates=[Sweden, Germany, Estonia, Spain, Israel, Austria, Ireland, Lithuania, Moldova, Czech_Republic]),
Ballot(ranked_candidates=[Israel, Denmark, Ireland, Cyprus, United_Kingdom, Moldova, Finland, Czech_Republic, Germany, Norway]),
Ballot(ranked_candidates=[Israel, Germany, Sweden, Albania, Bulgaria, Slovenia, Ireland, Czech_Republic, Estonia, The_Netherlands]),
Ballot(ranked_candidates=[Czech_Republic, Italy, Serbia, Israel, Germany, Denmark, Ireland, Hungary, Albania, Cyprus]),
Ballot(ranked_candidates=[Albania, Serbia, Hungary, Moldova, Ukraine, Bulgaria, Cyprus, Estonia, Germany, Israel]),
Ballot(ranked_candidates=[Israel, Cyprus, Ukraine, Norway, Czech_Republic, Italy, Moldova, Germany, Sweden, Bulgaria]),
Ballot(ranked_candidates=[Cyprus, Austria, Norway, Albania, Czech_Republic, Lithuania, Slovenia, Denmark, Australia, Sweden]),
Ballot(ranked_candidates=[Ukraine, Norway, Israel, Denmark, Moldova, Czech_Republic, Italy, Cyprus, Hungary, Bulgaria]),
Ballot(ranked_candidates=[Austria, The_Netherlands, Sweden, Czech_Republic, Israel, Germany, Cyprus, France, Albania, Spain]),
Ballot(ranked_candidates=[The_Netherlands, Israel, France, Cyprus, Italy, Denmark, Ireland, Austria, Germany, Czech_Republic]),
Ballot(ranked_candidates=[Austria, Lithuania, Czech_Republic, Albania, Germany, France, Israel, Cyprus, Sweden, Ireland]),
Ballot(ranked_candidates=[Cyprus, Israel, Moldova, Serbia, Italy, Hungary, Austria, Czech_Republic, Estonia, Ukraine]),
Ballot(ranked_candidates=[Lithuania, Israel, Moldova, Bulgaria, France, Norway, Czech_Republic, Ireland, United_Kingdom, Albania]),
Ballot(ranked_candidates=[Serbia, Italy, Cyprus, Slovenia, Israel, Czech_Republic, Albania, Germany, Denmark, Bulgaria]),
Ballot(ranked_candidates=[Sweden, Moldova, Italy, Spain, Albania, Czech_Republic, Hungary, France, Austria, Estonia]),
Ballot(ranked_candidates=[Bulgaria, Israel, Czech_Republic, Italy, Lithuania, Norway, Estonia, Germany, Ukraine, France]),
Ballot(ranked_candidates=[Israel, Ireland, Sweden, Slovenia, Albania, Austria, Bulgaria, Finland, Hungary, Lithuania]),
Ballot(ranked_candidates=[Ukraine, Israel, Cyprus, Denmark, Moldova, Estonia, Ireland, Germany, Italy, Bulgaria]),
Ballot(ranked_candidates=[Germany, Australia, Austria, Estonia, Spain, Cyprus, Sweden, Israel, France, Bulgaria]),
Ballot(ranked_candidates=[Germany, Austria, Norway, Sweden, Czech_Republic, The_Netherlands, Ireland, United_Kingdom, Australia, Cyprus]),
Ballot(ranked_candidates=[Austria, Lithuania, Cyprus, Bulgaria, Germany, Sweden, Norway, Portugal, Finland, The_Netherlands]),
Ballot(ranked_candidates=[Lithuania, Finland, Denmark, Italy, Austria, Czech_Republic, Cyprus, Hungary, Norway, Moldova]),
Ballot(ranked_candidates=[Estonia, Cyprus, Serbia, Sweden, Albania, Finland, Czech_Republic, Germany, Slovenia, Israel]),
Ballot(ranked_candidates=[Albania, Serbia, Cyprus, Bulgaria, Italy, Czech_Republic, Ukraine, Israel, Slovenia, Moldova]),
Ballot(ranked_candidates=[Israel, Bulgaria, Sweden, Austria, Cyprus, France, Italy, Denmark, Ireland, Germany]),
Ballot(ranked_candidates=[Estonia, Denmark, Hungary, Israel, Italy, Czech_Republic, France, Austria, The_Netherlands, Cyprus]),
Ballot(ranked_candidates=[Israel, Australia, Germany, Austria, The_Netherlands, Sweden, Czech_Republic, United_Kingdom, Slovenia, Ireland]),
Ballot(ranked_candidates=[Israel, Italy, Portugal, Estonia, Moldova, Spain, Ukraine, Cyprus, Denmark, Serbia]),
Ballot(ranked_candidates=[Sweden, Estonia, Austria, Germany, Bulgaria, The_Netherlands, Albania, Israel, Lithuania, Denmark]),
Ballot(ranked_candidates=[Israel, Cyprus, Ukraine, Lithuania, Estonia, Italy, France, Moldova, Denmark, Austria]),
Ballot(ranked_candidates=[Sweden, Austria, Ireland, Australia, Spain, The_Netherlands, Lithuania, Cyprus, Estonia, Israel]),
Ballot(ranked_candidates=[Italy, Israel, Czech_Republic, Ireland, Cyprus, Austria, Lithuania, Denmark, Hungary, United_Kingdom]),
Ballot(ranked_candidates=[Cyprus, Moldova, Sweden, Albania, Hungary, France, Israel, Estonia, Serbia, Italy]),
Ballot(ranked_candidates=[Cyprus, Albania, Italy, Czech_Republic, Israel, Bulgaria, Estonia, Norway, Germany, Moldova]),
Ballot(ranked_candidates=[Denmark, Albania, Austria, Australia, Israel, Lithuania, The_Netherlands, Ireland, Norway, Sweden]),
Ballot(ranked_candidates=[Denmark, Israel, Czech_Republic, Cyprus, Italy, Norway, The_Netherlands, Austria, Moldova, Germany]),
Ballot(ranked_candidates=[Austria, Albania, Israel, Estonia, Germany, Sweden, Ireland, France, Cyprus, Czech_Republic]),
Ballot(ranked_candidates=[Denmark, Czech_Republic, Germany, Israel, France, Austria, Norway, Finland, Sweden, Cyprus]),
Ballot(ranked_candidates=[Cyprus, Bulgaria, Germany, Israel, Portugal, Austria, France, Estonia, Albania, Spain]),
Ballot(ranked_candidates=[Lithuania, United_Kingdom, Germany, Czech_Republic, Israel, Cyprus, Estonia, The_Netherlands, Denmark, Moldova]),
Ballot(ranked_candidates=[Austria, Sweden, United_Kingdom, Australia, Finland, Germany, Slovenia, Hungary, Denmark, Bulgaria]),
Ballot(ranked_candidates=[Czech_Republic, Moldova, Estonia, Ukraine, France, Italy, Denmark, Germany, Cyprus, Austria]),
Ballot(ranked_candidates=[Norway, Germany, Denmark, Austria, United_Kingdom, Ireland, Estonia, Cyprus, Israel, Sweden]),
Ballot(ranked_candidates=[Albania, Moldova, Ukraine, Israel, Estonia, Cyprus, Denmark, Germany, Hungary, Serbia]),
Ballot(ranked_candidates=[Sweden, France, Estonia, Austria, Australia, The_Netherlands, Lithuania, Germany, United_Kingdom, Slovenia]),
Ballot(ranked_candidates=[Lithuania, Estonia, Israel, Moldova, Italy, Denmark, Czech_Republic, Ukraine, Norway, Cyprus]),
Ballot(ranked_candidates=[Austria, France, Sweden, Portugal, Israel, Germany, Ireland, The_Netherlands, Hungary, Cyprus]),
Ballot(ranked_candidates=[Estonia, Denmark, Czech_Republic, Italy, Cyprus, France, Sweden, Austria, Germany, Israel]),
Ballot(ranked_candidates=[Cyprus, Italy, France, Sweden, Israel, Czech_Republic, Norway, Australia, Bulgaria, Austria]),
Ballot(ranked_candidates=[Italy, Cyprus, Israel, Bulgaria, Australia, Lithuania, Germany, Czech_Republic, Denmark, United_Kingdom]),
Ballot(ranked_candidates=[Estonia, Israel, Bulgaria, Australia, Cyprus, Ukraine, Germany, Austria, Norway, The_Netherlands]),
Ballot(ranked_candidates=[Israel, Ukraine, Italy, Cyprus, Czech_Republic, Norway, Germany, Bulgaria, Hungary, Estonia]),
Ballot(ranked_candidates=[Serbia, Albania, Moldova, Norway, Denmark, Estonia, Italy, The_Netherlands, United_Kingdom, Cyprus]),
Ballot(ranked_candidates=[Serbia, Albania, Italy, Ukraine, Slovenia, Cyprus, Bulgaria, Hungary, Sweden, Israel]),
Ballot(ranked_candidates=[Germany, Sweden, Austria, France, Australia, Cyprus, The_Netherlands, Lithuania, Spain, Ireland]),
Ballot(ranked_candidates=[Lithuania, Denmark, Austria, Israel, Germany, The_Netherlands, Czech_Republic, Sweden, Cyprus, France]),
Ballot(ranked_candidates=[Austria, Germany, The_Netherlands, Albania, Sweden, Ireland, Australia, Hungary, Bulgaria, Lithuania]),
Ballot(ranked_candidates=[Ukraine, Israel, Cyprus, Hungary, Denmark, Italy, Czech_Republic, France, Ireland, Germany]),
Ballot(ranked_candidates=[Estonia, Albania, Austria, Bulgaria, Lithuania, France, Italy, Slovenia, Spain, Israel]),
Ballot(ranked_candidates=[Spain, Italy, Germany, Estonia, Moldova, Cyprus, Ukraine, Ireland, Denmark, Israel]),
Ballot(ranked_candidates=[Austria, Spain, The_Netherlands, Moldova, Lithuania, Cyprus, Germany, Estonia, Australia, Slovenia]),
Ballot(ranked_candidates=[Moldova, Hungary, Israel, Cyprus, Italy, Bulgaria, Germany, Czech_Republic, Denmark, Ireland]),
Ballot(ranked_candidates=[Moldova, Sweden, Israel, Cyprus, Estonia, Australia, Germany, Lithuania, France, Italy]),
Ballot(ranked_candidates=[Moldova, Israel, Ukraine, Denmark, Italy, Norway, Cyprus, Estonia, Czech_Republic, Sweden]),
Ballot(ranked_candidates=[Israel, Germany, Sweden, Moldova, Slovenia, Estonia, Italy, Serbia, Australia, Ireland]),
Ballot(ranked_candidates=[Israel, Czech_Republic, Italy, Cyprus, Denmark, Ukraine, Germany, Ireland, Moldova, United_Kingdom]),
Ballot(ranked_candidates=[Sweden, Germany, Italy, The_Netherlands, Norway, Moldova, Austria, Estonia, Israel, Albania]),
Ballot(ranked_candidates=[Hungary, Cyprus, Slovenia, Israel, Italy, Czech_Republic, Denmark, Norway, Bulgaria, Moldova]),
Ballot(ranked_candidates=[Sweden, Austria, Cyprus, The_Netherlands, Moldova, Estonia, Albania, Czech_Republic, France, Israel]),
Ballot(ranked_candidates=[Serbia, Italy, Czech_Republic, Denmark, Cyprus, Norway, Germany, Hungary, Austria, Albania]),
Ballot(ranked_candidates=[Cyprus, Israel, Austria, Germany, France, Ireland, Czech_Republic, Italy, Sweden, Estonia]),
Ballot(ranked_candidates=[Israel, Czech_Republic, Cyprus, Italy, Germany, Bulgaria, France, Norway, Estonia, Ireland]),
Ballot(ranked_candidates=[Cyprus, Austria, Australia, Israel, Bulgaria, France, Ireland, Finland, Norway, Germany]),
Ballot(ranked_candidates=[Denmark, Israel, Norway, Lithuania, Finland, Germany, Cyprus, Czech_Republic, Austria, The_Netherlands]),
Ballot(ranked_candidates=[Germany, Estonia, Lithuania, Cyprus, Ireland, Sweden, Austria, Portugal, Bulgaria, Israel]),
Ballot(ranked_candidates=[Serbia, Portugal, Italy, Albania, Germany, Israel, Austria, Cyprus, Denmark, Spain]),
Ballot(ranked_candidates=[Germany, Austria, Sweden, Lithuania, Cyprus, Israel, Estonia, Norway, Portugal, Slovenia]),
Ballot(ranked_candidates=[Germany, Israel, Denmark, Italy, Czech_Republic, Cyprus, Ireland, Austria, Hungary, Norway]),
Ballot(ranked_candidates=[France, Israel, The_Netherlands, Austria, Sweden, Slovenia, Czech_Republic, Denmark, Australia, Estonia]),
Ballot(ranked_candidates=[Israel, Czech_Republic, Denmark, France, Moldova, Italy, Cyprus, Estonia, Lithuania, Hungary]),
Ballot(ranked_candidates=[Austria, Israel, Bulgaria, Albania, Estonia, Norway, Finland, Ireland, Sweden, Spain]),
Ballot(ranked_candidates=[Lithuania, Ireland, Cyprus, Israel, Bulgaria, Czech_Republic, Moldova, Germany, Denmark, Australia]),
]
ballots_2019 = [Ballot(ranked_candidates=[North_Macedonia, Switzerland, Azerbaijan, Cyprus, Sweden, Italy, Greece, France, Australia, Russia]),
Ballot(ranked_candidates=[Sweden, North_Macedonia, Italy, Switzerland, The_Netherlands, Russia, Malta, Czech_Republic, United_Kingdom, Belarus]),
Ballot(ranked_candidates=[Sweden, France, Iceland, North_Macedonia, The_Netherlands, Czech_Republic, Russia, Germany, Azerbaijan, Serbia]),
Ballot(ranked_candidates=[North_Macedonia, Switzerland, The_Netherlands, Italy, Sweden, France, Azerbaijan, Czech_Republic, Serbia, Norway]),
Ballot(ranked_candidates=[Russia, Malta, North_Macedonia, Albania, Greece, Italy, Slovenia, Cyprus, Australia, Belarus]),
Ballot(ranked_candidates=[Malta, North_Macedonia, Cyprus, Italy, The_Netherlands, Azerbaijan, Switzerland, Greece, Sweden, Russia]),
Ballot(ranked_candidates=[Italy, Iceland, France, Switzerland, The_Netherlands, Czech_Republic, Australia, Greece, Cyprus, Denmark]),
Ballot(ranked_candidates=[Italy, North_Macedonia, Switzerland, Czech_Republic, The_Netherlands, Sweden, Serbia, France, Russia, Estonia]),
Ballot(ranked_candidates=[Greece, Russia, Italy, Sweden, Azerbaijan, The_Netherlands, France, Malta, Albania, Switzerland]),
Ballot(ranked_candidates=[Sweden, Slovenia, Italy, North_Macedonia, The_Netherlands, Azerbaijan, Iceland, Malta, France, Estonia]),
Ballot(ranked_candidates=[Sweden, North_Macedonia, Germany, The_Netherlands, Switzerland, Norway, Azerbaijan, Russia, Estonia, Italy]),
Ballot(ranked_candidates=[Sweden, Switzerland, Czech_Republic, The_Netherlands, Russia, Azerbaijan, Serbia, France, Denmark, Belarus]),
Ballot(ranked_candidates=[Sweden, Australia, The_Netherlands, North_Macedonia, Slovenia, Italy, Czech_Republic, Switzerland, Azerbaijan, Malta]),
Ballot(ranked_candidates=[The_Netherlands, Sweden, Italy, North_Macedonia, Azerbaijan, Iceland, Australia, Czech_Republic, Switzerland, Denmark]),
Ballot(ranked_candidates=[Czech_Republic, Azerbaijan, The_Netherlands, Denmark, Cyprus, France, Slovenia, Switzerland, Sweden, United_Kingdom]),
Ballot(ranked_candidates=[Italy, Australia, The_Netherlands, North_Macedonia, Switzerland, Norway, France, Malta, Sweden, Denmark]),
Ballot(ranked_candidates=[Cyprus, Russia, Azerbaijan, Italy, San_Marino, Malta, Slovenia, Albania, Australia, North_Macedonia]),
Ballot(ranked_candidates=[Czech_Republic, North_Macedonia, Belarus, Italy, Denmark, Azerbaijan, Sweden, Switzerland, United_Kingdom, The_Netherlands]),
Ballot(ranked_candidates=[Sweden, Australia, North_Macedonia, The_Netherlands, Czech_Republic, Switzerland, Azerbaijan, Italy, Serbia, France]),
Ballot(ranked_candidates=[Sweden, Switzerland, The_Netherlands, Azerbaijan, Norway, North_Macedonia, Australia, Russia, Germany, Italy]),
Ballot(ranked_candidates=[The_Netherlands, Italy, Estonia, Azerbaijan, Sweden, Australia, Malta, Switzerland, France, Czech_Republic]),
Ballot(ranked_candidates=[Denmark, North_Macedonia, Malta, Azerbaijan, Australia, Estonia, Czech_Republic, France, Sweden, Albania]),
Ballot(ranked_candidates=[The_Netherlands, Sweden, North_Macedonia, Denmark, Azerbaijan, Estonia, Russia, Czech_Republic, Italy, Malta]),
Ballot(ranked_candidates=[The_Netherlands, Azerbaijan, Sweden, Czech_Republic, Iceland, Germany, Slovenia, Italy, Australia, Russia]),
Ballot(ranked_candidates=[Italy, Russia, Azerbaijan, The_Netherlands, Cyprus, Sweden, Greece, North_Macedonia, Albania, San_Marino]),
Ballot(ranked_candidates=[North_Macedonia, Australia, Czech_Republic, Norway, Azerbaijan, Russia, Denmark, The_Netherlands, Sweden, Malta]),
Ballot(ranked_candidates=[Serbia, Russia, Albania, North_Macedonia, Malta, Cyprus, Australia, Denmark, Italy, Czech_Republic]),
Ballot(ranked_candidates=[Sweden, Switzerland, Malta, Denmark, Italy, Russia, Czech_Republic, North_Macedonia, Australia, Cyprus]),
Ballot(ranked_candidates=[Italy, Australia, Albania, The_Netherlands, Russia, Malta, Azerbaijan, Switzerland, Iceland, Cyprus]),
Ballot(ranked_candidates=[Czech_Republic, North_Macedonia, Sweden, The_Netherlands, Switzerland, Azerbaijan, Denmark, Italy, United_Kingdom, Cyprus]),
Ballot(ranked_candidates=[Australia, Slovenia, North_Macedonia, Serbia, France, Denmark, Malta, Iceland, Azerbaijan, Norway]),
Ballot(ranked_candidates=[The_Netherlands, Czech_Republic, Azerbaijan, Australia, Italy, North_Macedonia, Norway, Slovenia, Sweden, Switzerland]),
Ballot(ranked_candidates=[Australia, Azerbaijan, Czech_Republic, North_Macedonia, Russia, The_Netherlands, Switzerland, Serbia, Albania, Sweden]),
Ballot(ranked_candidates=[Azerbaijan, Greece, Cyprus, Belarus, Malta, San_Marino, North_Macedonia, Albania, Iceland, Spain]),
Ballot(ranked_candidates=[Italy, Russia, Greece, Albania, Iceland, Cyprus, Azerbaijan, The_Netherlands, Switzerland, North_Macedonia]),
Ballot(ranked_candidates=[North_Macedonia, Italy, Sweden, Australia, Estonia, Switzerland, Czech_Republic, Azerbaijan, Denmark, France]),
Ballot(ranked_candidates=[Czech_Republic, Azerbaijan, Italy, Sweden, The_Netherlands, Switzerland, Denmark, France, North_Macedonia, Malta]),
Ballot(ranked_candidates=[Sweden, Australia, The_Netherlands, Azerbaijan, Czech_Republic, Cyprus, Italy, Switzerland, Russia, Slovenia]),
Ballot(ranked_candidates=[The_Netherlands, Switzerland, Italy, Cyprus, Australia, Azerbaijan, Norway, Russia, Malta, Czech_Republic]),
Ballot(ranked_candidates=[North_Macedonia, The_Netherlands, Sweden, Norway, Germany, Italy, Australia, Russia, France, United_Kingdom]),
Ballot(ranked_candidates=[North_Macedonia, Sweden, Australia, Azerbaijan, The_Netherlands, Switzerland, Russia, Denmark, France, Czech_Republic]),
Ballot(ranked_candidates=[Russia, San_Marino, Italy, The_Netherlands, North_Macedonia, Norway, Switzerland, Azerbaijan, Greece, Australia]),
Ballot(ranked_candidates=[Russia, The_Netherlands, Switzerland, Italy, Malta, Norway, France, Iceland, Sweden, Australia]),
Ballot(ranked_candidates=[Norway, Iceland, Sweden, Switzerland, The_Netherlands, Italy, Malta, France, Czech_Republic, Azerbaijan]),
Ballot(ranked_candidates=[Switzerland, Italy, Norway, The_Netherlands, Iceland, Russia, Serbia, Australia, Slovenia, Azerbaijan]),
Ballot(ranked_candidates=[Russia, San_Marino, Switzerland, The_Netherlands, Italy, Belarus, Malta, North_Macedonia, Spain, Norway]),
Ballot(ranked_candidates=[Russia, The_Netherlands, Norway, Iceland, Azerbaijan, Slovenia, Switzerland, Italy, Australia, San_Marino]),
Ballot(ranked_candidates=[The_Netherlands, France, Italy, Norway, Spain, Switzerland, Australia, Iceland, Sweden, Russia]),
Ballot(ranked_candidates=[Italy, Slovenia, Serbia, North_Macedonia, Switzerland, Norway, The_Netherlands, Iceland, San_Marino, Albania]),
Ballot(ranked_candidates=[Greece, Russia, Italy, Switzerland, The_Netherlands, Israel, Spain, France, Australia, Norway]),
Ballot(ranked_candidates=[Russia, Norway, Australia, Azerbaijan, Iceland, Switzerland, The_Netherlands, Israel, Italy, Estonia]),
Ballot(ranked_candidates=[Norway, Sweden, Estonia, The_Netherlands, Switzerland, Azerbaijan, Iceland, Italy, Australia, Spain]),
Ballot(ranked_candidates=[Russia, Norway, The_Netherlands, Slovenia, Denmark, Iceland, Switzerland, Sweden, Australia, Azerbaijan]),
Ballot(ranked_candidates=[Iceland, Norway, Estonia, Sweden, Australia, The_Netherlands, Russia, Italy, Switzerland, Slovenia]),
Ballot(ranked_candidates=[Israel, Italy, Norway, Spain, Australia, The_Netherlands, Denmark, Russia, Switzerland, Iceland]),
Ballot(ranked_candidates=[Cyprus, San_Marino, Russia, Azerbaijan, Sweden, The_Netherlands, Israel, Iceland, North_Macedonia, Italy]),
Ballot(ranked_candidates=[Norway, Switzerland, Russia, The_Netherlands, Italy, Australia, Denmark, Slovenia, Iceland, Sweden]),
Ballot(ranked_candidates=[Cyprus, Italy, Russia, Switzerland, The_Netherlands, Albania, Spain, Australia, Iceland, France]),
Ballot(ranked_candidates=[Iceland, Norway, The_Netherlands, Russia, San_Marino, Switzerland, Italy, Slovenia, Azerbaijan, France]),
Ballot(ranked_candidates=[Norway, Australia, Sweden, Switzerland, Italy, The_Netherlands, Denmark, Estonia, Czech_Republic, San_Marino]),
Ballot(ranked_candidates=[Norway, Australia, The_Netherlands, Switzerland, Iceland, Russia, Italy, United_Kingdom, Sweden, Estonia]),
Ballot(ranked_candidates=[Russia, Norway, Italy, Switzerland, Australia, Spain, France, Azerbaijan, The_Netherlands, Denmark]),
Ballot(ranked_candidates=[Albania, Norway, Russia, Iceland, Australia, The_Netherlands, Denmark, Switzerland, France, Azerbaijan]),
Ballot(ranked_candidates=[Russia, Estonia, Norway, Iceland, Australia, The_Netherlands, Azerbaijan, Italy, Slovenia, Switzerland]),
Ballot(ranked_candidates=[Russia, Italy, Norway, The_Netherlands, Iceland, Azerbaijan, Estonia, Sweden, Switzerland, Australia]),
Ballot(ranked_candidates=[Italy, The_Netherlands, Switzerland, Norway, Sweden, North_Macedonia, Russia, Australia, Azerbaijan, Iceland]),
Ballot(ranked_candidates=[Russia, Azerbaijan, San_Marino, Israel, The_Netherlands, Italy, Switzerland, Norway, Czech_Republic, Iceland]),
Ballot(ranked_candidates=[Serbia, Russia, San_Marino, Albania, North_Macedonia, Italy, Slovenia, Azerbaijan, Iceland, The_Netherlands]),
Ballot(ranked_candidates=[Norway, Italy, Sweden, Iceland, Switzerland, Denmark, Azerbaijan, Spain, Australia, North_Macedonia]),
Ballot(ranked_candidates=[Albania, Serbia, San_Marino, The_Netherlands, Malta, Norway, Switzerland, Italy, Slovenia, Azerbaijan]),
Ballot(ranked_candidates=[Sweden, Iceland, The_Netherlands, Italy, Switzerland, Denmark, Australia, Azerbaijan, Estonia, Russia]),
Ballot(ranked_candidates=[Iceland, The_Netherlands, Norway, Italy, Australia, Switzerland, Slovenia, Russia, Azerbaijan, Czech_Republic]),
Ballot(ranked_candidates=[Spain, Russia, The_Netherlands, Italy, Norway, Switzerland, Australia, Iceland, France, Denmark]),
Ballot(ranked_candidates=[The_Netherlands, Switzerland, Italy, Russia, Azerbaijan, Iceland, Norway, Israel, Australia, France]),
Ballot(ranked_candidates=[Azerbaijan, Norway, Belarus, Iceland, Slovenia, The_Netherlands, Australia, Serbia, Switzerland, Italy]),
Ballot(ranked_candidates=[Russia, Greece, Italy, Cyprus, The_Netherlands, Switzerland, Azerbaijan, Norway, Iceland, Israel]),
Ballot(ranked_candidates=[North_Macedonia, Slovenia, Russia, Italy, Switzerland, Iceland, Norway, The_Netherlands, Spain, San_Marino]),
Ballot(ranked_candidates=[North_Macedonia, Serbia, Italy, Iceland, Norway, The_Netherlands, Switzerland, Denmark, Australia, Estonia]),
Ballot(ranked_candidates=[Italy, Switzerland, The_Netherlands, Norway, Sweden, Australia, France, Iceland, Russia, Azerbaijan]),
Ballot(ranked_candidates=[Norway, Estonia, Iceland, Denmark, The_Netherlands, Australia, Italy, Azerbaijan, North_Macedonia, Switzerland]),
Ballot(ranked_candidates=[Italy, Albania, Norway, Serbia, The_Netherlands, Spain, Sweden, France, North_Macedonia, Denmark]),
Ballot(ranked_candidates=[Norway, Australia, Iceland, Switzerland, Denmark, Sweden, The_Netherlands, Azerbaijan, Spain, Cyprus]),
]
ballots_2021 = [Ballot(ranked_candidates=[Switzerland, France, Malta, Cyprus, Greece, San_Marino, Italy, Finland, Azerbaijan, Serbia]),
Ballot(ranked_candidates=[Malta, Switzerland, Iceland, France, Italy, Ukraine, Cyprus, The_Netherlands, Bulgaria, Russia]),
Ballot(ranked_candidates=[Iceland, Switzerland, France, Portugal, Italy, Bulgaria, Malta, The_Netherlands, Germany, Finland]),
Ballot(ranked_candidates=[Russia, Greece, Moldova, Malta, Ukraine, Sweden, France, Belgium, Portugal, Bulgaria]),
Ballot(ranked_candidates=[Switzerland, Ukraine, Finland, Russia, Bulgaria, Malta, Sweden, Iceland, Italy, Portugal]),
Ballot(ranked_candidates=[Moldova, Italy, Greece, France, Portugal, Malta, Spain, Israel, The_Netherlands, Finland]),
Ballot(ranked_candidates=[Italy, Iceland, Switzerland, Serbia, France, Israel, Russia, Malta, Lithuania, Portugal]),
Ballot(ranked_candidates=[Greece, Malta, Italy, France, Russia, Bulgaria, Belgium, Finland, Switzerland, Portugal]),
Ballot(ranked_candidates=[Portugal, France, Iceland, Malta, Italy, Switzerland, Bulgaria, Belgium, Russia, Finland]),
Ballot(ranked_candidates=[Switzerland, Iceland, Finland, Albania, Bulgaria, San_Marino, Malta, Norway, Sweden, Israel]),
Ballot(ranked_candidates=[Switzerland, France, Iceland, Finland, Bulgaria, Portugal, Ukraine, Italy, Lithuania, Malta]),
Ballot(ranked_candidates=[Switzerland, Bulgaria, Iceland, France, Italy, Sweden, Russia, Lithuania, Greece, Malta]),
Ballot(ranked_candidates=[Greece, Russia, Portugal, Switzerland, Belgium, Israel, San_Marino, Iceland, Cyprus, Sweden]),
Ballot(ranked_candidates=[Italy, Switzerland, Portugal, Ukraine, Iceland, Azerbaijan, Bulgaria, Lithuania, The_Netherlands, Malta]),
Ballot(ranked_candidates=[France, Switzerland, Malta, Cyprus, Italy, Ukraine, Sweden, Iceland, Russia, Lithuania]),
Ballot(ranked_candidates=[Cyprus, Moldova, Bulgaria, San_Marino, Malta, France, Italy, Belgium, Russia, Ukraine]),
Ballot(ranked_candidates=[Switzerland, Portugal, Bulgaria, Italy, France, Finland, Greece, Ukraine, Russia, Malta]),
Ballot(ranked_candidates=[France, Malta, Iceland, Norway, Ukraine, Switzerland, Lithuania, Belgium, Azerbaijan, Bulgaria]),
Ballot(ranked_candidates=[Switzerland, Lithuania, France, Russia, Belgium, Malta, Ukraine, Cyprus, Azerbaijan, Bulgaria]),
Ballot(ranked_candidates=[Lithuania, Finland, Iceland, Malta, Portugal, Belgium, Israel, France, Norway, Ukraine]),
Ballot(ranked_candidates=[Switzerland, Iceland, Italy, Ukraine, Lithuania, Bulgaria, Finland, France, Malta, Russia]),
Ballot(ranked_candidates=[Ukraine, Italy, Switzerland, Belgium, Portugal, France, Iceland, Malta, Bulgaria, Israel]),
Ballot(ranked_candidates=[Albania, Switzerland, Sweden, Portugal, Greece, Ukraine, Cyprus, France, Finland, San_Marino]),
Ballot(ranked_candidates=[Bulgaria, Russia, Greece, Malta, Azerbaijan, Iceland, France, Switzerland, Portugal, Cyprus]),
Ballot(ranked_candidates=[France, Iceland, Russia, Malta, Belgium, Switzerland, Israel, Ukraine, Azerbaijan, Albania]),
Ballot(ranked_candidates=[Serbia, Italy, Israel, France, Switzerland, Malta, Iceland, Sweden, Cyprus, Russia]),
Ballot(ranked_candidates=[Malta, Sweden, Israel, Switzerland, Bulgaria, Italy, France, Ukraine, Iceland, Greece]),
Ballot(ranked_candidates=[San_Marino, Iceland, Portugal, Switzerland, Israel, Italy, Malta, Belgium, Finland, Russia]),
Ballot(ranked_candidates=[Bulgaria, Russia, Iceland, Switzerland, France, Belgium, Malta, Italy, Ukraine, The_Netherlands]),
Ballot(ranked_candidates=[Malta, Portugal, Finland, Switzerland, Moldova, Ukraine, France, Italy, Bulgaria, Germany]),
Ballot(ranked_candidates=[Moldova, Italy, Azerbaijan, Greece, Bulgaria, Israel, Malta, Belgium, Cyprus, Switzerland]),
Ballot(ranked_candidates=[France, Italy, Greece, Malta, Lithuania, Moldova, Switzerland, Bulgaria, Albania, Finland]),
Ballot(ranked_candidates=[France, Finland, Italy, Iceland, Bulgaria, Portugal, Sweden, Greece, Israel, Switzerland]),
Ballot(ranked_candidates=[Italy, Iceland, Russia, Switzerland, Belgium, Malta, Finland, Greece, France, Israel]),
Ballot(ranked_candidates=[France, Switzerland, Malta, Iceland, Cyprus, Portugal, Bulgaria, Israel, Lithuania, Greece]),
Ballot(ranked_candidates=[Malta, Italy, Ukraine, Iceland, France, Switzerland, Israel, Russia, Norway, Belgium]),
Ballot(ranked_candidates=[France, Bulgaria, Italy, Portugal, Malta, Iceland, Lithuania, Russia, Azerbaijan, Finland]),
Ballot(ranked_candidates=[Italy, France, Switzerland, Israel, Belgium, Malta, Iceland, Azerbaijan, Portugal, Norway]),
Ballot(ranked_candidates=[France, Iceland, Switzerland, Portugal, Israel, Bulgaria, Finland, San_Marino, Spain, Belgium]),
Ballot(ranked_candidates=[Switzerland, Italy, Greece, Moldova, France, Bulgaria, Ukraine, Finland, Cyprus, Sweden]),
Ballot(ranked_candidates=[Iceland, Ukraine, Malta, Italy, Lithuania, Switzerland, France, Finland, Serbia, Norway]),
Ballot(ranked_candidates=[Serbia, Iceland, Italy, Ukraine, France, Switzerland, Finland, Lithuania, Malta, Norway]),
Ballot(ranked_candidates=[Israel, Italy, Ukraine, Norway, Russia, Finland, Switzerland, San_Marino, France, Moldova]),
Ballot(ranked_candidates=[France, Ukraine, Italy, Lithuania, Finland, Iceland, Switzerland, Sweden, Malta, Norway]),
Ballot(ranked_candidates=[Italy, France, Finland, Russia, Ukraine, Serbia, Azerbaijan, Switzerland, Greece, Sweden]),
Ballot(ranked_candidates=[Serbia, Italy, Ukraine, France, Finland, Switzerland, Iceland, Russia, Azerbaijan, Albania]),
Ballot(ranked_candidates=[Greece, Italy, France, Bulgaria, Ukraine, Lithuania, Finland, Russia, Israel, Malta]),
Ballot(ranked_candidates=[Moldova, Ukraine, Greece, Iceland, Italy, Russia, Finland, Switzerland, France, Lithuania]),
Ballot(ranked_candidates=[Iceland, Sweden, Norway, Finland, Switzerland, Italy, Lithuania, France, Malta, Ukraine]),
Ballot(ranked_candidates=[Finland, Lithuania, Italy, France, Russia, Ukraine, Norway, Iceland, Switzerland, Sweden]),
Ballot(ranked_candidates=[Iceland, Ukraine, Italy, Switzerland, France, Lithuania, Norway, Sweden, Moldova, Russia]),
Ballot(ranked_candidates=[Ukraine, Italy, Portugal, Moldova, Switzerland, Israel, Iceland, Serbia, Sweden, Finland]),
Ballot(ranked_candidates=[Greece, Lithuania, Italy, San_Marino, Ukraine, France, Russia, Azerbaijan, Moldova, Iceland]),
Ballot(ranked_candidates=[Lithuania, France, Finland, Italy, Iceland, Russia, Ukraine, Serbia, Switzerland, Norway]),
Ballot(ranked_candidates=[Cyprus, Italy, France, Albania, Finland, Ukraine, Switzerland, Malta, Azerbaijan, Russia]),
Ballot(ranked_candidates=[Finland, Sweden, Ukraine, France, Switzerland, Italy, Lithuania, Malta, Portugal, Norway]),
Ballot(ranked_candidates=[Lithuania, Iceland, Ukraine, France, Italy, Finland, Malta, Switzerland, Portugal, Sweden]),
Ballot(ranked_candidates=[Ukraine, Russia, France, Italy, Switzerland, Malta, Finland, Lithuania, Azerbaijan, Iceland]),
Ballot(ranked_candidates=[Ukraine, Albania, Finland, Russia, Iceland, Lithuania, Serbia, San_Marino, Moldova, France]),
Ballot(ranked_candidates=[Lithuania, Russia, Finland, Italy, Ukraine, Iceland, Switzerland, France, Norway, Sweden]),
Ballot(ranked_candidates=[Ukraine, Italy, France, Finland, Switzerland, Norway, Russia, Iceland, Belgium, Sweden]),
Ballot(ranked_candidates=[Italy, Norway, Sweden, Finland, Lithuania, Iceland, Serbia, France, Cyprus, Ukraine]),
Ballot(ranked_candidates=[Russia, Ukraine, Italy, Greece, France, Finland, Switzerland, Sweden, Lithuania, Azerbaijan]),
Ballot(ranked_candidates=[France, Greece, Iceland, Switzerland, Portugal, Ukraine, Finland, Lithuania, Italy, Malta]),
Ballot(ranked_candidates=[Serbia, Albania, Italy, Switzerland, Cyprus, France, Ukraine, Azerbaijan, Finland, Russia]),
Ballot(ranked_candidates=[Lithuania, Iceland, Sweden, Italy, Finland, Ukraine, France, Malta, Switzerland, Azerbaijan]),
Ballot(ranked_candidates=[Ukraine, Italy, Iceland, Switzerland, Finland, France, Lithuania, Norway, Russia, Sweden]),
Ballot(ranked_candidates=[France, Ukraine, Moldova, Italy, Switzerland, Finland, Sweden, Iceland, Greece, Russia]),
Ballot(ranked_candidates=[Moldova, Italy, France, Ukraine, Finland, Switzerland, Azerbaijan, Lithuania, Malta, Israel]),
Ballot(ranked_candidates=[Cyprus, Italy, Finland, Ukraine, France, Switzerland, Azerbaijan, Moldova, Lithuania, Iceland]),
Ballot(ranked_candidates=[Italy, France, Cyprus, Greece, Moldova, Ukraine, Finland, Switzerland, Bulgaria, Russia]),
Ballot(ranked_candidates=[Italy, France, Ukraine, Finland, Russia, Iceland, Azerbaijan, Greece, Cyprus, Switzerland]),
Ballot(ranked_candidates=[Serbia, Italy, Greece, Ukraine, France, Switzerland, Finland, Iceland, Norway, Russia]),
Ballot(ranked_candidates=[France, Italy, Bulgaria, Switzerland, Ukraine, Iceland, Lithuania, Malta, Finland, Portugal]),
Ballot(ranked_candidates=[Finland, Iceland, Norway, Lithuania, France, Switzerland, Ukraine, Italy, Malta, Serbia]),
Ballot(ranked_candidates=[Serbia, Italy, Portugal, Albania, France, Iceland, Finland, Sweden, Ukraine, Lithuania]),
Ballot(ranked_candidates=[Italy, Lithuania, Finland, Switzerland, Iceland, France, Russia, Azerbaijan, Sweden, Belgium]),
Ballot(ranked_candidates=[Lithuania, Iceland, Bulgaria, Finland, Malta, France, Ukraine, Italy, Norway, Switzerland]),
]


election_result = pyrankvote.instant_runoff_voting(candidates, ballots_2021)
winners = election_result.get_winners()

br_2016 = bootstrap_replicates_pr(ballots_2016, replicates=1000)
br_2016.to_csv('Results/AV_Bootstrap_results_2016.csv')

br_2017 = bootstrap_replicates_pr(ballots_2017, replicates=1000)
br_2017.to_csv('Results/AV_Bootstrap_results_2017.csv')

br_2018 = bootstrap_replicates_pr(ballots_2018, replicates=1000)
br_2018.to_csv('Results/AV_Bootstrap_results_2018.csv')

br_2019 = bootstrap_replicates_pr(ballots_2019, replicates=1000)
br_2019.to_csv('Results/AV_Bootstrap_results_2019.csv')

# Scrap

df.groupby(['Year'])['PV'].sum().reset_index()

# End of Code
