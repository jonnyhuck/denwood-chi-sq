"""
* This is Chi Squared Goodness of Fit test - comparing one distribution with another
* https://stats.stackexchange.com/questions/110718/chi-squared-test-with-scipy-whats-the-difference-between-chi2-contingency-and/375063
*
* https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
* https://colorswall.com/images/palettes/bootstrap-4-colors-3-colorswall.png
*
* Result:
*    observed: [ 6  7 30 66]
*    expected: [ 6 10 32 64]
*    X2= 1.0875 p= 0.7800924582345854
*    ----------
*    Cannot Reject H0, difference not significant (p=0.780092)
"""

from numpy import array
from math import ceil, floor
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from pandas import read_csv, IntervalIndex, cut, value_counts


# read in MLL dataset
years = read_csv("./data/years.csv")

# get extreme values
min_year = int(years.All.min())
max_year = int(years.All.max())
n_years = max_year - min_year

# create bins
bins = IntervalIndex.from_tuples([(1979, 2005), (2005, 2010), (2010, 2015), (2015, 2021)])

observed = value_counts(cut(years.Sample, bins), sort=False, ascending=True).to_numpy()
expected = value_counts(cut(years.All, bins), sort=False, ascending=True).to_numpy()

# scale to get expected values
expected = array([ ceil((x / sum(expected) * sum(observed))) for x in expected])

# get chi2 and p values
chi2, p = chisquare(observed, f_exp=expected)

# output to console
print("")
print("observed:", observed)
print("expected:", expected)
print("X2=", chi2)
print("p=", p)
print("----------")
if (p < 0.05):
    print(f"Reject H0: The distributions do not match (p={p:.6f})")
else:
    print(f"Cannot Reject H0, difference not significant (p={p:.6f})")
print("")

''' PLOT '''

# init plot
fig, axes = plt.subplots(figsize=(15, 8), nrows=1, ncols=3)

# create bin labels
labels = array(['1979-2005', '2005-2010', '2010-2015', '2015-2020'])

# observed plot
plt.subplot(131)
plt.bar(labels, observed, width=0.8, align='center', color="#5bc0de")
plt.xlabel('Year of Publication')
plt.ylabel('Frequency')
plt.ylim([0, 70])
plt.title('Observed Values')

# expected plot
plt.subplot(132)
plt.bar(labels, expected, width=0.8, align='center', color="#f0ad4e")
plt.xlabel('Year of Publication')
plt.ylabel('Frequency')
plt.ylim([0, 70])
plt.title('Expected Values')

# calculate difference (balance) for third plot
difference = observed - expected

# difference plot
plt.subplot(133)
mask1 = difference <= 0
mask2 = difference > 0
plt.bar(labels[mask1], difference[mask1], width=0.8, align='center', color="#d9534f")
plt.bar(labels[mask2], difference[mask2], width=0.8, align='center', color="#0275d8")
plt.xlabel('Year of Publication')
plt.ylabel('Frequency')
plt.ylim([-70, 70])
plt.axhline(y=0, linewidth=0.5, color='k')
plt.title('Difference')

# output image
plt.savefig('./out/year.png', dpi=300)
