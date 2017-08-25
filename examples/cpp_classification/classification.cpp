#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  std::vector<std::vector<Prediction> > Classify(const std::vector<cv::Mat>& imgs, int N = 5);

 private:
  void SetMean(const string& mean_file);

  std::vector<std::vector<float> > Predict(const std::vector<cv::Mat>& imgs);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels, int n);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  // [aaronn] commenting out check below because GoogleNet 
  // has many outputs
  //CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<std::vector<Prediction> > Classifier::Classify(const std::vector<cv::Mat>& imgs, int N) {
  std::vector<std::vector<float> > outputs = Predict(imgs);

  std::vector<std::vector<Prediction> > all_predictions;
  for (int j = 0; j < outputs.size(); ++j) {
    std::vector<float> output = outputs[j];

    N = std::min<int>(labels_.size(), N);
    std::vector<int> maxN = Argmax(output, N);
    std::vector<Prediction> predictions;
    for (int i = 0; i < N; ++i) {
      int idx = maxN[i];
      predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }
    all_predictions.push_back(predictions);
 }

  return all_predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<std::vector<float> > Classifier::Predict(const std::vector<cv::Mat>& imgs) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(imgs.size(), num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  for (int i = 0; i < imgs.size(); ++i) {
      std::vector<cv::Mat> input_channels;
      WrapInputLayer(&input_channels, i);
      Preprocess(imgs[i], &input_channels);
      //net_->dump_input_blob();

  }

  net_->ForwardPrefilled();
  //net_->dump_blob("data", true);

  std::vector<std::vector<float> > outputs;

  /* TODO pick best output from googlenet (which has 3 outputs) */
  Blob<float>* output_layer 
  //= net_->output_blobs()[net_->output_blobs().size()-1];
    = net_->output_blobs()[0];
  for (int i = 0; i < output_layer->num(); ++i) {
      const float* begin = output_layer->cpu_data() + i * output_layer->channels();
      const float* end = begin + output_layer->channels();
      /* Copy the output layer to a std::vector */
      outputs.push_back(std::vector<float>(begin, end));
  }

  //net_->dump_blob( "pool5", true);
  //net_->dump_blob( "fc6", false);

  return outputs;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels, int n) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  int channels = input_layer->channels();
  float* input_data = input_layer->mutable_cpu_data() + n * width * height * channels;
  for (int i = 0; i < channels; ++i) {
      cv::Mat channel(height, width, CV_32FC1, input_data);
      input_channels->push_back(channel);
      input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

/*
  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
*/
}

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  Classifier classifier(model_file, trained_file, mean_file, label_file);

  std::vector<cv::Mat> imgsFromUser;
  std::vector<string> imgPathsFromUser;
  int batchMultiplier = 0;
  for (int i = 5; i < argc; ++i)
  {
      batchMultiplier = atoi(argv[i]);
      if (batchMultiplier > 0)
      {
        std::cout << "ANDBG batchMultiplier: " << batchMultiplier << std::endl;
        break;
      }

      cv::Mat img = cv::imread(argv[i], -1);
      CHECK(!img.empty()) << "Unable to decode image " << argv[i];
      imgsFromUser.push_back(img);
      imgPathsFromUser.push_back(argv[i]);
  }

  batchMultiplier = std::max(batchMultiplier, 1);
  std::vector<cv::Mat> imgs;
  std::vector<string> imgPaths;
  for (int b=0; b < batchMultiplier; b++)
    for (int i=0; i < imgsFromUser.size(); i++)
    {
      imgs.push_back(imgsFromUser[i]);
      imgPaths.push_back(imgPathsFromUser[i]);
    }

  std::vector<std::vector<Prediction> > all_predictions = classifier.Classify(imgs);

  /* Print the top N predictions. */
  for (size_t i = 0; i < all_predictions.size(); ++i) {
    std::cout << "---------- Prediction " << i << " for "
              << imgPaths[i] << " ----------" << std::endl;

    std::vector<Prediction>& predictions = all_predictions[i];
    for (size_t j = 0; j < predictions.size(); ++j) {
        Prediction p = predictions[j];
        std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                  << p.first << "\"" << std::endl;
    }
    std::cout << std::endl;
  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
