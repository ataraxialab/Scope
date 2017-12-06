/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure
with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a
single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to
multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV
2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented
Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but
more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with
fixed_window = true)

Default values are set for all properties of the tracker depending on the above
choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added
stability

For speed, the value (template_size/cell_size) should be a power of 2 or a
product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this
license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#ifndef _KCFTRACKER_HEADERS
#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "fhog.hpp"
#include "recttools.hpp"
#endif

KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multi_scale) {
  // Parameters equal in all cases
  lambda = 0.0001;
  padding = 2.5;
  // output_sigma_factor = 0.1;
  output_sigma_factor = 0.125;

  if (hog) {
    // VOT
    interp_factor = 0.012;
    sigma = 0.6;
    // TPAMI
    // interp_factor = 0.02;
    // sigma = 0.5;
    cell_size = 4;
    _hogfeatures = true;
  } else {
    interp_factor = 0.075;
    sigma = 0.2;
    cell_size = 1;
    _hogfeatures = false;
  }

  if (multi_scale) {
    template_size = 96;
    // template_size = 100;
    scale_step = 1.05;
    scale_weight = 0.95;
    if (!fixed_window) {
      fixed_window = true;
    }
  } else if (fixed_window) {
    template_size = 96;
    // template_size = 100;
    scale_step = 1;
  } else {
    template_size = 1;
    scale_step = 1;
  }
}

void KCFTracker::init(const cv::Rect &roi, const cv::Mat &im_mat) {
  _roi = roi;
  assert(roi.width >= 0 && roi.height >= 0);
  _tmpl = getFeatures(im_mat, 1);
  _prob = createGaussianPeak(size_patch[0], size_patch[1]);
  _alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
  train(_tmpl, 1.0);
}

void KCFTracker::update(const cv::Mat &im_mat, const cv::Rect &roi) {
  _roi.x = roi.x;
  _roi.y = roi.y;
  _roi.width = roi.width;
  _roi.height = roi.height;

  if (_roi.x >= im_mat.cols - 1) _roi.x = im_mat.cols - 1;
  if (_roi.y >= im_mat.rows - 1) _roi.y = im_mat.rows - 1;
  if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
  if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

  assert(_roi.width >= 0 && _roi.height >= 0);
  cv::Mat x = getFeatures(im_mat, 0);
  train(x, 3 * interp_factor);
}

cv::Rect KCFTracker::tracking(const cv::Mat &im_mat) {
  if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
  if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
  if (_roi.x >= im_mat.cols - 1) _roi.x = im_mat.cols - 2;
  if (_roi.y >= im_mat.rows - 1) _roi.y = im_mat.rows - 2;

  float cx = _roi.x + _roi.width / 2.0f;
  float cy = _roi.y + _roi.height / 2.0f;

  cv::Point2f res = detect(_tmpl, getFeatures(im_mat, 0, 1.0f), score_);

  if (scale_step != 1) {
    float new_score;
    cv::Point2f new_res =
        detect(_tmpl, getFeatures(im_mat, 0, 1.0f / scale_step), new_score);

    if (scale_weight * new_score > score_) {
      res = new_res;
      score_ = new_score;
      _scale /= scale_step;
      _roi.width /= scale_step;
      _roi.height /= scale_step;
    }

    new_res = detect(_tmpl, getFeatures(im_mat, 0, scale_step), new_score);

    if (scale_weight * new_score > score_) {
      res = new_res;
      score_ = new_score;
      _scale *= scale_step;
      _roi.width *= scale_step;
      _roi.height *= scale_step;
    }
  }

  // Adjust by cell size and _scale
  _roi.x = cx - _roi.width / 2.0f + (res.x * cell_size * _scale);
  _roi.y = cy - _roi.height / 2.0f + (res.y * cell_size * _scale);

  if (_roi.x >= im_mat.cols - 1) _roi.x = im_mat.cols - 1;
  if (_roi.y >= im_mat.rows - 1) _roi.y = im_mat.rows - 1;
  if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
  if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

  assert(_roi.width >= 0 && _roi.height >= 0);
  cv::Mat x = getFeatures(im_mat, 0);
  train(x, interp_factor);

  return _roi;
}

cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value) {
  using namespace FFTTools;

  cv::Mat k = gaussianCorrelation(x, z);
  cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));

  cv::Point2i pi;
  double pv;
  cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
  peak_value = (float)pv;

  cv::Point2f p((float)pi.x, (float)pi.y);

  if (pi.x > 0 && pi.x < res.cols - 1) {
    p.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value,
                        res.at<float>(pi.y, pi.x + 1));
  }

  if (pi.y > 0 && pi.y < res.rows - 1) {
    p.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value,
                        res.at<float>(pi.y + 1, pi.x));
  }

  p.x -= (res.cols) / 2;
  p.y -= (res.rows) / 2;

  return p;
}

void KCFTracker::train(cv::Mat x, float train_interp_factor) {
  using namespace FFTTools;

  cv::Mat k = gaussianCorrelation(x, x);
  cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));

  _tmpl = (1 - train_interp_factor) * _tmpl + train_interp_factor * x;
  _alphaf = (1 - train_interp_factor) * _alphaf + train_interp_factor * alphaf;
}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts
// between input images X and Y, which must both be MxN. They must    also be
// periodic (ie., pre-processed with a cosine window).
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2) {
  using namespace FFTTools;
  cv::Mat c =
      cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));
  if (_hogfeatures) {
    cv::Mat caux;
    cv::Mat x1aux;
    cv::Mat x2aux;
    for (int i = 0; i < size_patch[2]; i++) {
      x1aux = x1.row(i);  // Procedure do deal with cv::Mat multichannel bug
      x1aux = x1aux.reshape(1, size_patch[0]);
      x2aux = x2.row(i).reshape(1, size_patch[0]);
      cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
      caux = fftd(caux, true);
      rearrange(caux);
      caux.convertTo(caux, CV_32F);
      c = c + real(caux);
    }
  } else {
    cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
    c = fftd(c, true);
    rearrange(c);
    c = real(c);
  }
  cv::Mat d;
  cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c) /
              (size_patch[0] * size_patch[1] * size_patch[2]),
          0, d);

  cv::Mat k;
  cv::exp((-d / (sigma * sigma)), k);
  return k;
}

cv::Mat KCFTracker::createGaussianPeak(int size_y, int size_x) {
  cv::Mat_<float> res(size_y, size_x);

  int syh = size_y / 2;
  int sxh = size_x / 2;

  float output_sigma = std::sqrt(static_cast<float>(size_x * size_y)) /
                       padding * output_sigma_factor;
  float mult = -0.5f / (output_sigma * output_sigma);

  for (int i = 0; i < size_y; i++)
    for (int j = 0; j < size_x; j++) {
      int ih = i - syh;
      int jh = j - sxh;
      res(i, j) = std::exp(mult * (ih * ih + jh * jh));
    }
  return FFTTools::fftd(res);
}

cv::Mat KCFTracker::getFeatures(const cv::Mat &im_mat, bool inithann,
                                float scale_adjust) {
  float cx = _roi.x + _roi.width / 2.f;
  float cy = _roi.y + _roi.height / 2.f;

  if (inithann) {
    float padded_w = _roi.width * padding;
    float padded_h = _roi.height * padding;

    if (template_size > 1) {
      // Fit largest dimension to the given template size
      if (padded_w >= padded_h) {
        _scale = padded_w / template_size;
      } else {
        _scale = padded_h / template_size;
      }
      _tmpl_sz.width = static_cast<int>(padded_w / _scale);
      _tmpl_sz.height = static_cast<int>(padded_h / _scale);
    } else {
      // No template size given, use ROI size
      _tmpl_sz.width = static_cast<int>(padded_w);
      _tmpl_sz.height = static_cast<int>(padded_h);
      _scale = 1;
    }

    if (_hogfeatures) {
      int even = 2 * cell_size;
      _tmpl_sz.width = (_tmpl_sz.width / even + 1) * even;
      _tmpl_sz.height = (_tmpl_sz.height / even + 1) * even;
    } else {
      _tmpl_sz.width = (_tmpl_sz.width >> 1) << 1;
      _tmpl_sz.height = (_tmpl_sz.height >> 1) << 1;
    }
  }

  cv::Rect scale_roi;
  scale_roi.width = static_cast<int>(scale_adjust * _scale * _tmpl_sz.width);
  scale_roi.height = static_cast<int>(scale_adjust * _scale * _tmpl_sz.height);
  scale_roi.x = static_cast<int>(cx - scale_roi.width / 2.f);
  scale_roi.y = static_cast<int>(cy - scale_roi.height / 2.f);

  cv::Mat im_roi =
      RectTools::subwindow(im_mat, scale_roi, cv::BORDER_REPLICATE);

  if (im_roi.cols != _tmpl_sz.width || im_roi.rows != _tmpl_sz.height) {
    cv::resize(im_roi, im_roi, _tmpl_sz);
  }

  cv::Mat FeaturesMap;
  if (_hogfeatures) {
    CvLSVMFeatureMapCaskade *map;
    getFeatureMaps(im_roi, cell_size, &map);
    normalizeAndTruncate(map, 0.2f);
    PCAFeatureMaps(map);

    size_patch[0] = map->sizeY;
    size_patch[1] = map->sizeX;
    size_patch[2] = map->numFeatures;

    FeaturesMap =
        cv::Mat(cv::Size(map->numFeatures, map->sizeX * map->sizeY), CV_32F,
                map->map);  // Procedure do deal with cv::Mat multichannel bug
    FeaturesMap = FeaturesMap.t();
    freeFeatureMapObject(&map);
  } else {
    FeaturesMap = RectTools::getGrayImage(im_roi);
    FeaturesMap -= 0.5f;
    size_patch[0] = im_roi.rows;
    size_patch[1] = im_roi.cols;
    size_patch[2] = 1;
  }

  if (inithann) {
    createHanningMats();
  }
  FeaturesMap = hann.mul(FeaturesMap);

  return FeaturesMap;
}

void KCFTracker::createHanningMats() {
  int size_y = size_patch[0], size_x = size_patch[1], num_fea = size_patch[2];

  cv::Mat hann1t = cv::Mat(cv::Size(size_x, 1), CV_32F, cv::Scalar(0));
  cv::Mat hann2t = cv::Mat(cv::Size(1, size_y), CV_32F, cv::Scalar(0));

  for (int i = 0; i < size_x; i++) {
    hann1t.at<float>(0, i) =
        0.5f * (1 - std::cos(2 * 3.14159265358979323846f * i / (size_x - 1)));
  }
  for (int i = 0; i < size_y; i++) {
    hann2t.at<float>(i, 0) =
        0.5f * (1 - std::cos(2 * 3.14159265358979323846f * i / (size_y - 1)));
  }
  cv::Mat hann2d = hann2t * hann1t;
  if (_hogfeatures) {
    cv::Mat hann1d = hann2d.reshape(1, 1);

    hann = cv::Mat(cv::Size(size_y * size_x, num_fea), CV_32F, cv::Scalar(0));
    for (int i = 0; i < num_fea; i++) {
      for (int j = 0; j < size_y * size_x; ++j) {
        hann.at<float>(i, j) = hann1d.at<float>(0, j);
      }
    }
  } else {
    hann = hann2d;
  }
}

float KCFTracker::subPixelPeak(float left, float center, float right) {
  float divisor = 2 * center - right - left;

  if (divisor == 0) return 0;

  return 0.5f * (right - left) / divisor;
}
