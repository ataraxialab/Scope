#ifndef ATLAB_SCOPE_H
#define ATLAB_SCOPE_H

#if defined(_WIN32)
#define SCOPE_EXPORTS
#endif

#ifdef SCOPE_EXPORTS
#define SCOPE_API __declspec(dllexport)
#else
#define SCOPE_API
#endif


#ifndef __cplusplus
extern "C" {
#endif

typedef void* MHandle;
typedef MHandle SCOPE_ENGINE;
typedef int MInt32;
typedef MInt32  MRESULT;

typedef struct __tag_rect
{
  MInt32 left;
  MInt32 top;
  MInt32 right;
  MInt32 bottom;
} MRECT, *PMRECT;



typedef struct {
  /// Number of objects
  MInt32 num_objects;
  /// Pointer to objects' rectangles, should be initialized by user.
  MRECT *objects;
  /// Pointer to objects' labels, 0 for pedestrian and 1 for vehicle, should be
  /// initialized by user.(Pedestrian is unfinished.)
  MInt32 *labels;
} SCOPE_OUTPUT, *PTR_SCOPE_OUTPUT;

typedef struct {
  /// Number of objects
  MInt32 idx;
  std::string FileName;
  MInt32 FrameIdx;
  /// Pointer to objects' rectangles, should be initialized by user.
  MRECT BBox;
  float score;
  /// Pointer to objects' labels, 0 for pedestrian and 1 for vehicle, should be
  /// initialized by user.(Pedestrian is unfinished.)
  MInt32 labels;
} SAVE_INFO, *PTR_SAVE_INFO;


/**
 * The function is used to initialize the SCOPE engine.
 *
 * @param  scope_engine [OUT] Pointer to the handle of a SCOPE engine.
 *
 * @return MRESULT    [OUT] Return MOK if success, otherwise failed.
 */
SCOPE_API MRESULT Scope_Initial(SCOPE_ENGINE *scope_engine);


#if defined(USE_OpenCV)
#include "opencv2/opencv.hpp"
/**
 * The function is used to detect vehicles and pedestrians.
 *
 * @param  vpd_engine [IN] Handle to the SCOPE engine.
 * @param  img_data   [IN] input image data.
 * @param  roi        [IN] Region of interest to detect.
 * @param  vpd_output [OUT] Pointer to the SCOPE_OUTPUT struct.
 *
 * @return MRESULT    [OUT] Return MOK if success, otherwise failed.
 */
SCOPE_API MRESULT Scope_Detect(SCOPE_ENGINE scope_engine, const cv::Mat &img_data,
                               PTR_SCOPE_OUTPUT scope_output);
#endif

SCOPE_API MRESULT Scope_WriteDB(SCOPE_ENGINE scope_engine,
                                std::vector<SAVE_INFO> save_infos);
/**
 * The function is used to release the SCOPE engine.
 *
 * @param  vpd_engine [IN] Handle to the SCOPE engine.
 *
 * @return MRESULT    [OUT] Return MOK if success, otherwise failed.
 */
SCOPE_API MRESULT Scope_Release(SCOPE_ENGINE scope_engine);

#ifndef __cplusplus
}
#endif

#endif  // ATLAB_SCOPE_H
