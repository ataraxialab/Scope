#include "fhog.hpp"

int getFeatureMaps(const cv::Mat &image, int k, CvLSVMFeatureMapCaskade **map) {
  int channel = image.channels(), height = image.rows, width = image.cols;

  float kernel[3] = {-1.f, 0.f, 1.f};
  cv::Mat kernel_dx(1, 3, CV_32F, kernel), kernel_dy(3, 1, CV_32F, kernel);
  cv::Mat dx(height, width, CV_32FC3), dy(height, width, CV_32FC3);

  cv::filter2D(image, dx, dx.depth(), kernel_dx, cv::Point(-1, 0));
  cv::filter2D(image, dy, dx.depth(), kernel_dy, cv::Point(0, -1));

  int size_x = width / k, size_y = height / k;
  int num_features = 3 * NUM_SECTOR, map_step = size_x * num_features;
  allocFeatureMapObject(map, size_x, size_y, num_features);

  float cos_bin[NUM_SECTOR];
  float sin_bin[NUM_SECTOR];
  for (int n = 0; n < NUM_SECTOR; ++n) {
    float bin_val = n * static_cast<float>(PI) / NUM_SECTOR;
    cos_bin[n] = std::cos(bin_val);
    sin_bin[n] = std::sin(bin_val);
  }

  std::vector<float> magnitude(height * width);
  std::vector<int> alpha(height * width * 2);
  for (int h = 1; h < height - 1; ++h) {
    float *dx_index = (float *)(dx.data + dx.step * h);
    float *dy_index = (float *)(dy.data + dy.step * h);
    for (int w = 1; w < width - 1; ++w) {
      int mag_idx = h * width + w, alpha_idx = mag_idx * 2;

      float mag_val = -FLT_MAX, dx_val = 0, dy_val = 0;
      for (int c = 0; c < channel; ++c) {
        float tx = dx_index[w * channel + c];
        float ty = dy_index[w * channel + c];
        float mag = std::sqrt(tx * tx + ty * ty);
        if (mag > mag_val) {
          mag_val = mag;
          dx_val = tx;
          dy_val = ty;
        }
      }
      magnitude[mag_idx] = mag_val;

      float prod_val = cos_bin[0] * dx_val + sin_bin[0] * dy_val;
      int max_idx = 0;
      for (int n = 1; n < NUM_SECTOR; ++n) {
        float prod = cos_bin[n] * dx_val + sin_bin[n] * dy_val;
        if (prod > prod_val) {
          prod_val = prod;
          max_idx = n;
        } else if (-prod > prod_val) {
          prod_val = -prod;
          max_idx = n + NUM_SECTOR;
        }
      }
      alpha[alpha_idx] = max_idx % NUM_SECTOR;
      alpha[alpha_idx + 1] = max_idx;
    }
  }

  std::vector<int> near(k);
  for (int i = 0; i < k / 2; ++i) {
    near[i] = -1;
  }
  for (int i = k / 2; i < k; ++i) {
    near[i] = 1;
  }

  std::vector<float> weight(k * 2);
  for (int i = 0; i < k / 2; ++i) {
    float a_x = k / 2 - i - 0.5f;
    float b_x = k / 2 + i + 0.5f;
    weight[i * 2] = 1.0f / a_x * ((a_x * b_x) / (a_x + b_x));
    weight[i * 2 + 1] = 1.0f / b_x * ((a_x * b_x) / (a_x + b_x));
  }
  for (int i = k / 2; i < k; ++i) {
    float a_x = i - k / 2 + 0.5f;
    float b_x = -i + k / 2 - 0.5f + k;
    weight[i * 2] = 1.0f / a_x * ((a_x * b_x) / (a_x + b_x));
    weight[i * 2 + 1] = 1.0f / b_x * ((a_x * b_x) / (a_x + b_x));
  }

  float *map_data = (*map)->map;
  for (int h = 1; h < height - 1; ++h) {
    for (int w = 1; w < width - 1; ++w) {
      int i = h / k, j = w / k, ki = h % k, kj = w % k;

      float w_i0 = weight[ki * 2], w_i1 = weight[ki * 2 + 1];
      float w_j0 = weight[kj * 2], w_j1 = weight[kj * 2 + 1];

      int mag_idx = h * width + w, alpha_idx = mag_idx * 2;
      float mag_val = magnitude[mag_idx];

      int map_off = i * map_step + j * num_features;
      float mag = mag_val * w_i0 * w_j0;
      map_data[map_off + alpha[alpha_idx]] += mag;
      map_data[map_off + alpha[alpha_idx + 1] + NUM_SECTOR] += mag;

      int near_i = i + near[ki], near_j = j + near[kj];
      if (near_i >= 0 && near_i < size_y) {
        map_off = near_i * map_step + j * num_features;
        mag = mag_val * w_i1 * w_j0;
        map_data[map_off + alpha[alpha_idx]] += mag;
        map_data[map_off + alpha[alpha_idx + 1] + NUM_SECTOR] += mag;
      }

      if (near_j >= 0 && near_j < size_x) {
        map_off = i * map_step + near_j * num_features;
        mag = mag_val * w_i0 * w_j1;
        map_data[map_off + alpha[alpha_idx]] += mag;
        map_data[map_off + alpha[alpha_idx + 1] + NUM_SECTOR] += mag;
      }

      if (near_i >= 0 && near_i < size_y && near_j >= 0 && near_j < size_x) {
        map_off = near_i * map_step + near_j * num_features;
        mag = mag_val * w_i1 * w_j1;
        map_data[map_off + alpha[alpha_idx]] += mag;
        map_data[map_off + alpha[alpha_idx + 1] + NUM_SECTOR] += mag;
      }
    }
  }

  return LATENT_SVM_OK;
}

int normalizeAndTruncate(CvLSVMFeatureMapCaskade *map, float alpha) {
  int origin_x = map->sizeX, origin_y = map->sizeY;
  int size_x = origin_x - 2, size_y = origin_y - 2;

  int num_fea = map->numFeatures, map_step = origin_x * num_fea;
  int new_num_fea = 12 * NUM_SECTOR, new_map_step = size_x * new_num_fea;

  int p = NUM_SECTOR;

  std::vector<float> sum_squares(origin_x * origin_y);
  for (int i = 0; i < origin_x * origin_y; ++i) {
    float sum = 0.0f;
    int offset = i * num_fea;
    for (int j = 0; j < NUM_SECTOR; ++j) {
      sum += map->map[offset + j] * map->map[offset + j];
    }
    sum_squares[i] = sum;
  }

  float *new_data = new float[size_x * size_y * new_num_fea];

  for (int i = 1; i < origin_y - 1; ++i) {
    for (int j = 1; j < origin_x - 1; ++j) {
      int ori_off = i * map_step + j * num_fea;
      int new_off = (i - 1) * new_map_step + (j - 1) * new_num_fea;

      float norm = std::sqrt(sum_squares[i * origin_x + j] +
                             sum_squares[i * origin_x + j + 1] +
                             sum_squares[(i + 1) * origin_x + j] +
                             sum_squares[(i + 1) * origin_x + j + 1]) +
                   FLT_EPSILON;
      for (int n = 0; n < p; ++n) {
        new_data[new_off + n] = map->map[ori_off + n] / norm;
      }
      for (int n = 0; n < 2 * p; ++n) {
        new_data[new_off + p * 4 + n] = map->map[ori_off + p + n] / norm;
      }

      norm = std::sqrt(sum_squares[i * origin_x + j] +
                       sum_squares[i * origin_x + j + 1] +
                       sum_squares[(i - 1) * origin_x + j] +
                       sum_squares[(i - 1) * origin_x + j + 1]) +
             FLT_EPSILON;
      for (int n = 0; n < p; ++n) {
        new_data[new_off + p + n] = map->map[ori_off + n] / norm;
      }
      for (int n = 0; n < 2 * p; ++n) {
        new_data[new_off + p * 6 + n] = map->map[ori_off + p + n] / norm;
      }

      norm = std::sqrt(sum_squares[i * origin_x + j] +
                       sum_squares[i * origin_x + j - 1] +
                       sum_squares[(i + 1) * origin_x + j] +
                       sum_squares[(i + 1) * origin_x + j - 1]) +
             FLT_EPSILON;
      for (int n = 0; n < p; ++n) {
        new_data[new_off + p * 2 + n] = map->map[ori_off + n] / norm;
      }
      for (int n = 0; n < 2 * p; n++) {
        new_data[new_off + p * 8 + n] = map->map[ori_off + p + n] / norm;
      }

      norm = std::sqrt(sum_squares[i * origin_x + j] +
                       sum_squares[i * origin_x + j - 1] +
                       sum_squares[(i - 1) * origin_x + j] +
                       sum_squares[(i - 1) * origin_x + j - 1]) +
             FLT_EPSILON;
      for (int n = 0; n < p; ++n) {
        new_data[new_off + p * 3 + n] = map->map[ori_off + n] / norm;
      }
      for (int n = 0; n < 2 * p; ++n) {
        new_data[new_off + p * 10 + n] = map->map[ori_off + p + n] / norm;
      }
    }
  }

  for (int i = 0; i < size_x * size_y * new_num_fea; ++i) {
    if (new_data[i] > alpha) new_data[i] = alpha;
  }

  map->numFeatures = new_num_fea;
  map->sizeX = size_x;
  map->sizeY = size_y;

  delete[] map->map;

  map->map = new_data;

  return LATENT_SVM_OK;
}

int PCAFeatureMaps(CvLSVMFeatureMapCaskade *map) {
  int size_x = map->sizeX, size_y = map->sizeY;
  int num_fea = map->numFeatures, map_step = size_x * num_fea;
  int new_num_fea = 3 * NUM_SECTOR + 4, new_map_step = size_x * new_num_fea;
  int corner = 4;
  int p = NUM_SECTOR;

  float nx = 1.0f / std::sqrt(static_cast<float>(p * 2));
  float ny = 1.0f / std::sqrt(static_cast<float>(corner));

  float *new_data = new float[size_x * size_y * new_num_fea];

  for (int i = 0; i < size_y; ++i) {
    for (int j = 0; j < size_x; ++j) {
      int ori_off = i * map_step + j * num_fea;
      int new_off = i * new_map_step + j * new_num_fea;

      int k = 0;
      for (int n = 0; n < p * 2; ++n) {
        float sum = 0;
        for (int c = 0; c < corner; ++c) {
          sum += map->map[ori_off + corner * p + c * p * 2 + n];
        }
        new_data[new_off + k] = sum * ny;
        k++;
      }

      for (int n = 0; n < p; ++n) {
        float sum = 0;
        for (int c = 0; c < corner; ++c) {
          sum += map->map[ori_off + c * p + n];
        }
        new_data[new_off + k] = sum * ny;
        k++;
      }

      for (int c = 0; c < corner; ++c) {
        float sum = 0;
        for (int n = 0; n < 2 * p; ++n) {
          sum += map->map[ori_off + corner * p + c * p * 2 + n];
        }
        new_data[new_off + k] = sum * nx;
        k++;
      }
    }
  }

  map->numFeatures = new_num_fea;

  delete[] map->map;

  map->map = new_data;

  return LATENT_SVM_OK;
}

int allocFeatureMapObject(CvLSVMFeatureMapCaskade **obj, int size_x, int size_y,
                          int num_features) {
  *obj = new CvLSVMFeatureMapCaskade;
  (*obj)->sizeX = size_x;
  (*obj)->sizeY = size_y;
  (*obj)->numFeatures = num_features;
  (*obj)->map = new float[size_x * size_y * num_features];
  for (int i = 0; i < size_x * size_y * num_features; ++i) {
    (*obj)->map[i] = 0.0f;
  }
  return LATENT_SVM_OK;
}

int freeFeatureMapObject(CvLSVMFeatureMapCaskade **obj) {
  if (*obj == NULL) return LATENT_SVM_MEM_NULL;
  delete[](*obj)->map;
  delete *obj;
  *obj = NULL;
  return LATENT_SVM_OK;
}
