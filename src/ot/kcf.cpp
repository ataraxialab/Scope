#include "kcf.hpp"

template <typename Dtype>
inline bool NearBorder(const Box<Dtype> &box, int width) {
  float gap = 0.02f * width;
  return (box.xmin < gap) || (box.xmax > width - gap);
  // float gap = 0.02f * 540;
  // return (box.xmin - 475 < gap) || (box.xmax - 475 > 950 - gap);
}

inline VecBoxF AnalyseHistory(const std::vector<VecBoxF> history_boxes,
                              float threshold) {
  const VecBoxF &boxes_0 = history_boxes[0], &boxes_1 = history_boxes[1];

  VecBoxF same_boxes;
  VecInt check_0(boxes_0.size(), 0), check_1(boxes_1.size(), 0);
  for (int i = 0; i < boxes_0.size(); ++i) {
    if (check_0[i]) continue;
    BoxF box_i = boxes_0[i];
    for (int j = 0; j < boxes_1.size(); ++j) {
      if (check_1[j]) continue;
      const BoxF &box_j = boxes_1[j];
      float iou = Boxes::IoU(box_i, box_j);
      if (iou > threshold) {
        check_0[i] = 1;
        check_1[j] = 1;
        float smooth = box_i.score / (box_i.score + box_j.score);
        Boxes::Smooth(box_j, &box_i, smooth);
        box_i.score = (box_i.score + box_j.score) / 2.f;
        same_boxes.push_back(box_i);
      }
    }
  }
  return same_boxes;
}

void OT::FeedOD(const cv::Mat &im_mat, const VecBoxF &od_boxes) {
  RemoveUntracking();

  VecInt check_od(od_boxes.size(), 0);
  for (auto &engine : engines_) {
    BoxF &box_ot = engine.box_;
    for (int i = 0; i < od_boxes.size(); ++i) {
      if (check_od[i]) continue;
      const BoxF &box_od = od_boxes[i];
      float iou = Boxes::IoU(box_od, box_ot);
      float in = Boxes::Intersection(box_od, box_ot);
      if (iou > 0.9) {
        check_od[i] = 1;
        engine.count_ = 0;
      } else if (iou > 0.6) {
        float smooth = box_ot.score / (box_ot.score + 1);
        Boxes::Smooth(box_od, &box_ot, smooth);
        UpdateRegion(im_mat, &engine);
        check_od[i] = 1;
        engine.count_ = 0;
      }
    }
  }

  history_count_ = ++history_count_ % history_period_;
  history_boxes_[history_count_].clear();
  for (int i = 0; i < od_boxes.size(); ++i) {
    if (check_od[i] == 1) continue;
    history_boxes_[history_count_].push_back(od_boxes[i]);
  }
  const VecBoxF &same_boxes = AnalyseHistory(history_boxes_, 0.7);
  SetupEngine(im_mat, same_boxes);
}

void OT::Tracking(const cv::Mat &im_mat, VecBoxF *ot_boxes,VecInt * ot_uids) {
  for (auto &it : engines_) {
    if (it.tracking_) {
      const cv::Rect &cv_roi = it.engine_.tracking(im_mat);
      it.box_.xmin = cv_roi.x;
      it.box_.ymin = cv_roi.y;
      it.box_.xmax = cv_roi.x + cv_roi.width;
      it.box_.ymax = cv_roi.y + cv_roi.height;
      it.box_.score = it.engine_.score_;
      it.count_++;
    }
  }

  for (auto en_i = engines_.begin(); en_i != engines_.end(); ++en_i) {
    BoxF &box_i = (*en_i).box_;
    if (box_i.label == -1 || !(*en_i).tracking_) continue;
    auto en_j = en_i;
    for (en_j++; en_j != engines_.end(); ++en_j) {
      BoxF &box_j = (*en_j).box_;
      if (box_j.label == -1 || box_i.label != box_j.label || !(*en_i).tracking_)
        continue;
      if (Boxes::IoU(box_i, box_j) > 0.5) {
        float score_i = box_i.score, score_j = box_j.score;
        float smooth = score_i / (score_i + score_j);
        Boxes::Smooth(box_j, &box_i, smooth);
        box_j.label = -1;
        continue;
      }
      float in = Boxes::Intersection(box_i, box_j);
      float cover_i = in / Boxes::Size(box_i);
      float cover_j = in / Boxes::Size(box_j);
      if (cover_i > cover_j && cover_i > 0.9) box_i.label = -1;
      if (cover_i < cover_j && cover_j > 0.9) box_j.label = -1;
    }
  }

  engines_.remove_if([](const OTTracter &engine) {
    if (engine.tracking_) {
      const BoxF &box = engine.box_;
      bool small_size =
          (box.xmax - box.xmin) < 30 || (box.ymax - box.ymin) < 20;
      bool wrong_ratio = (box.xmax - box.xmin) / (box.ymax - box.ymin) < 0.7;
      bool low_score = box.score < 0.4;
      bool always_track = engine.count_ > 90;
      return box.label == -1 || small_size || wrong_ratio || low_score ||
             always_track;
    } else {
      return false;
    }
  });

  ot_boxes->clear();
  ot_uids->clear();
  for (auto &engine : engines_) {
    if (engine.tracking_) {
      engine.tracking_ = 1;
    }
    ot_boxes->push_back(engine.box_);
    ot_uids->push_back(engine.uid);
  }
}

void OT::SetupEngine(const cv::Mat &im_mat, const BoxF &box) {
  if (TrackingNum() >= max_tracking_num_) {
    return;
  }
  const BoxI roi(box);
  cv::Rect cv_roi(roi.xmin, roi.ymin, roi.xmax - roi.xmin, roi.ymax - roi.ymin);

  OTTracter tracker;
  tracker.box_ = box;
  tracker.engine_.init(cv_roi, im_mat);
  tracker.tracking_ = true;
  tracker.uid = uid_idx;
  uid_idx++;

  /*
  if (!NearBorder(box, im_mat.cols)) {

  } else {
    tracker.tracking_ = false;
  }
  */
  engines_.push_back(tracker);
}

void OT::SetupEngine(const cv::Mat &im_mat, const VecBoxF &boxes) {
  for (const auto &box : boxes) {
    SetupEngine(im_mat, box);
  }
}

void OT::ReleaseEngine(std::list<OTTracter>::iterator it) {
  engines_.erase(it);
}

void OT::UpdateRegion(const cv::Mat &im_mat, OTTracter *it) {
  const BoxI roi((*it).box_);
  cv::Rect cv_roi(roi.xmin, roi.ymin, roi.xmax - roi.xmin, roi.ymax - roi.ymin);

  if ((*it).tracking_) {
    (*it).engine_.update(im_mat, cv_roi);
  } else if (!NearBorder(roi, im_mat.cols) &&
             TrackingNum() < max_tracking_num_) {
    (*it).engine_.init(cv_roi, im_mat);
    (*it).tracking_ = true;
  }
}

void OT::RemoveUntracking() {
  engines_.remove_if([](const OTTracter &engine) { return !engine.tracking_; });
}

int OT::TrackingNum() {
  int count = 0;
  for (const auto &engine : engines_) {
    if (engine.tracking_) count++;
  }
  return count;
}
