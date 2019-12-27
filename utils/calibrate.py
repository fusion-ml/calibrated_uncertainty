"""
For calibrating models and quantifying calibration
"""

import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
plt.ion()

def get_proportion_lists(residuals, stds, num_bins=100):
  exp_proportions = np.linspace(0, 1, num_bins)
  obs_proportions = [get_proportion_in_interval(quantile, residuals, stds) for
                     quantile in tqdm(exp_proportions, desc='Calibration')]
  return exp_proportions, obs_proportions

def get_proportion_in_interval(quantile, residuals, stds):
  norm = stats.norm(loc=0, scale=1)
  lower_bound = norm.ppf(0.5-quantile/2)
  upper_bound = norm.ppf(0.5+quantile/2)
  normalized_residuals = residuals.reshape(-1) / stds.reshape(-1)
  num_within_quantile = 0
  for resid in normalized_residuals:
    if lower_bound <= resid <= upper_bound:
      num_within_quantile += 1.
  proportion = num_within_quantile / len(residuals)
  return proportion

def plot_calibration_curve(exp_proportions, obs_proportions,
                           curve_label=None):

  # Set figure defaults
  width = 5
  fontsize = 12
  rc = {'figure.figsize': (width, width),
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'legend.fontsize': fontsize}
  sns.set(rc=rc)
  sns.set_style('ticks')

  # Set label 
  if curve_label is None:
    curve_label = 'Predictor'
  # Plot
  plt.figure()
  plt.plot([0, 1], [0, 1], '--', label='Ideal')
  plt.plot(exp_proportions, obs_proportions, label=curve_label)
  plt.fill_between(exp_proportions, exp_proportions, obs_proportions,
                   alpha=0.2)
  plt.xlabel('Expected proportion in interval')
  plt.ylabel('Observed proportion in interval')
  plt.axis('square')
  buff = 0.01
  plt.xlim([0-buff, 1+buff])
  plt.ylim([0-buff, 1+buff])

  # Compute miscalibration area
  polygon_points = []
  for point in zip(exp_proportions, obs_proportions):
      polygon_points.append(point)
  for point in zip(reversed(exp_proportions), reversed(exp_proportions)):
      polygon_points.append(point)
  polygon_points.append((exp_proportions[0], obs_proportions[0]))
  polygon = Polygon(polygon_points)
  x, y = polygon.exterior.xy # original data
  ls = LineString(np.c_[x, y]) # closed, non-simple
  lr = LineString(ls.coords[:] + ls.coords[0:1])
  mls = unary_union(lr)
  polygon_area_list =[poly.area for poly in polygonize(mls)]
  miscalibration_area = np.asarray(polygon_area_list).sum()

  # Annotate plot with the miscalibration area
  plt.text(x=0.95, y=0.05,
           s='Miscalibration area = %.2f' % miscalibration_area,
           verticalalignment='bottom',
           horizontalalignment='right',
           fontsize=fontsize)


def plot_residuals_vs_stds(residuals, stds):
  # Put stds on same scale as residuals
  res_sum = np.sum(np.abs(residuals))
  stds_scaled = (stds / np.sum(stds)) * res_sum 
  # Plot
  plt.figure()
  plt.plot(stds, np.abs(residuals),'x')
  lims = [np.min([plt.xlim()[0], plt.ylim()[0]]),
          np.max([plt.xlim()[1], plt.ylim()[1]])]
  plt.plot(lims, lims, '--', label='Ideal')
  plt.xlabel('Standard deviations (scaled)')
  plt.ylabel('Residuals (absolute value)')
  plt.axis('square')
  plt.xlim(lims)
  plt.ylim(lims)
