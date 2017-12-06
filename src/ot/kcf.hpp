#include "kcftracker.hpp"

#include "../common/boxes.hpp"

using namespace Scope;

class OTTracter {
 public:
  KCFTracker engine_;
  BoxF box_;
  int count_ = 0;
  bool tracking_ = false;
};

class OT {
 public:
  explicit OT(int max_tracking_num = 10) : max_tracking_num_(max_tracking_num) {
    history_boxes_.resize(history_period_);
  }
  ~OT() {
    engines_.clear();
    history_boxes_.clear();
  }

  void FeedOD(const cv::Mat &im_mat, const VecBoxF &od_boxes);

  void Tracking(const cv::Mat &im_mat, VecBoxF *ot_boxes);

 private:
  inline void SetupEngine(const cv::Mat &im_mat, const BoxF &box);
  inline void SetupEngine(const cv::Mat &im_mat, const VecBoxF &boxes);

  inline void ReleaseEngine(std::list<OTTracter>::iterator it);

  inline void UpdateRegion(const cv::Mat &im_mat, OTTracter *it);

  inline void RemoveUntracking();

  inline int TrackingNum();

  int max_tracking_num_;

  std::list<OTTracter> engines_;

  int history_count_ = -1, history_period_ = 2;
  std::vector<VecBoxF> history_boxes_;
};
