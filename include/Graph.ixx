module;
export module Graph;
import Pipeline;
import Texture;
import Swapchain;
import Sync;
import Timestamps;
import Command;
import CommandPool;
import Buffer;
import Device;
import Window;
import glm;
import <volk.h>;
import "BS_thread_pool.hpp";
import <map>;

export namespace RenderGraph {
class GraphStorage final {
 private:
  std::map<std::string, std::unique_ptr<ImageViewHolder>> _imageViewHolders;
  std::map<std::string, std::vector<std::unique_ptr<Buffer>>> _buffers;

 public:
  GraphStorage() = default;
  GraphStorage(const GraphStorage&) = delete;
  GraphStorage& operator=(const GraphStorage&) = delete;
  GraphStorage(GraphStorage&&) = delete;
  GraphStorage& operator=(GraphStorage&&) = delete;

  void add(std::string_view name, std::unique_ptr<ImageViewHolder> imageViewHolder) noexcept;
  // not const because will do std::move
  void add(std::string_view name, std::vector<std::unique_ptr<Buffer>>& buffers) noexcept;
  void reset(
             std::vector<std::shared_ptr<ImageView>> oldSwapchain,
             std::vector<std::shared_ptr<ImageView>> newSwapchain,
             const CommandBuffer& commandBuffer) noexcept;
  std::string find(const std::vector<std::shared_ptr<ImageView>>& imageViews) noexcept;
  const ImageViewHolder& getImageViewHolder(std::string_view name) const noexcept;
  // NVRO
  std::vector<Buffer*> getBuffer(std::string_view name) const noexcept;
};

class GraphElement {
 public:
  virtual void draw(int currentFrame, const CommandBuffer& commandBuffer) = 0;
  virtual void update(int currentFrame, const CommandBuffer& commandBuffer) = 0;
  virtual void reset(const std::vector<std::shared_ptr<RenderGraph::ImageView>>& swapchain,
                     const CommandBuffer& commandBuffer) = 0;
  virtual ~GraphElement() = default;
};

enum class GraphPassType { GRAPHIC, COMPUTE };

class GraphPass {
 protected:
  std::string _name;
  GraphPassType _graphPassType;
  const GraphStorage* _graphStorage;
  std::unique_ptr<CommandPool> _commandPool;
  std::vector<std::unique_ptr<CommandBuffer>> _commandBuffers;
  std::vector<std::pair<std::vector<std::shared_ptr<Semaphore>>, std::function<int()>>> _signalSemaphores,
      _waitSemaphores;
  std::vector<std::shared_ptr<GraphElement>> _graphElements;

 public:
  GraphPass(std::string_view name, GraphPassType graphPassType, const GraphStorage& graphStorage) noexcept;
  GraphPass(const GraphPass&) = delete;
  GraphPass& operator=(const GraphPass&) = delete;
  GraphPass(GraphPass&&) = delete;
  GraphPass& operator=(GraphPass&&) = delete;

  void registerGraphElement(std::shared_ptr<GraphElement> graphElement) noexcept;
  // not const because will do std::move
  void addSignalSemaphore(std::vector<std::shared_ptr<Semaphore>>& signalSemaphore,
                          std::function<int()> index) noexcept;
  void addWaitSemaphore(std::vector<std::shared_ptr<Semaphore>>& waitSemaphore, std::function<int()> index) noexcept;
  // NVRO
  GraphPassType getGraphPassType() const noexcept;
  std::vector<Semaphore*> getSignalSemaphores() const noexcept;
  std::vector<Semaphore*> getWaitSemaphores() const noexcept;
  std::vector<CommandBuffer*> getCommandBuffers() const noexcept;
  std::string getName() const noexcept;
  virtual void execute(int currentFrame, const CommandBuffer& commandBuffer) = 0;
  void reset(const std::vector<std::shared_ptr<RenderGraph::ImageView>>& swapchain, CommandBuffer& commandBuffer);
  virtual ~GraphPass() = default;
};

class GraphPassGraphic final : public GraphPass {
 private:
  std::vector<std::string> _colorTargets, _textureInputs;
  std::optional<std::string> _depthTarget;

  std::map<std::string, bool> _clearTarget;
  std::unique_ptr<PipelineGraphic> _pipelineGraphic;
  const Device* _device;

 public:
  GraphPassGraphic(std::string_view name,
                   int maxFramesInFlight,
                   const GraphStorage& graphStorage,
                   const Device& device) noexcept;
  GraphPassGraphic(const GraphPassGraphic&) = delete;
  GraphPassGraphic& operator=(const GraphPassGraphic&) = delete;
  GraphPassGraphic(GraphPassGraphic&&) = delete;
  GraphPassGraphic& operator=(GraphPassGraphic&&) = delete;

  // handle attachments
  void addColorTarget(std::string_view name) noexcept;
  void setDepthTarget(std::string_view name) noexcept;
  // handle input to shaders
  void addTextureInput(std::string_view name) noexcept;

  // clear color target before use or load
  void clearTarget(std::string_view name) noexcept;

  const std::vector<std::string>& getColorTargets() const noexcept;
  std::optional<std::string> getDepthTarget() const noexcept;
  const std::vector<std::string>& getTextureInputs() const noexcept;
  PipelineGraphic& getPipelineGraphic(const GraphStorage& graphStorage) const noexcept;
  void execute(int currentFrame, const CommandBuffer& commandBuffer) override;
};

class GraphPassCompute final : public GraphPass {
 private:
  std::vector<std::string> _storageBufferInputs, _storageBufferOutputs;
  std::vector<std::string> _storageTextureInputs, _storageTextureOutputs;
  bool _separate = false;
  const Device* _device;

 public:
  GraphPassCompute(std::string_view name,
                   int maxFramesInFlight,
                   bool separate,
                   const GraphStorage& graphStorage,
                   const Device& device) noexcept;
  GraphPassCompute(const GraphPassCompute&) = delete;
  GraphPassCompute& operator=(const GraphPassCompute&) = delete;
  GraphPassCompute(GraphPassCompute&&) = delete;
  GraphPassCompute& operator=(GraphPassCompute&&) = delete;

  // handle input to shaders
  void addStorageBufferInput(std::string_view name) noexcept;
  void addStorageTextureInput(std::string_view name) noexcept;
  // handle output from shaders
  void addStorageBufferOutput(std::string_view name) noexcept;
  void addStorageTextureOutput(std::string_view name) noexcept;

  const std::vector<std::string>& getStorageBufferInputs() const noexcept;
  const std::vector<std::string>& getStorageBufferOutputs() const noexcept;
  const std::vector<std::string>& getStorageTextureInputs() const noexcept;
  const std::vector<std::string>& getStorageTextureOutputs() const noexcept;
  bool isSeparate() const noexcept;
  void execute(int currentFrame, const CommandBuffer& commandBuffer) override;
};

class Graph final {
 private:
  Swapchain* _swapchain;
  const Device* _device;
  const Window* _window;
  std::unique_ptr<BS::thread_pool> _threadPool;
  std::vector<std::unique_ptr<GraphPass>> _passes;
  std::deque<GraphPass*> _passesOrdered;
  std::unique_ptr<Timestamps> _timestamps;
  std::unique_ptr<GraphStorage> _graphStorage;
  std::unique_ptr<CommandPool> _commandPoolReset;
  std::unique_ptr<CommandBuffer> _commandBuffersReset;
  bool _resetFrames;
  // special semaphores
  std::vector<std::shared_ptr<Semaphore>> _semaphoreRenderFinished, _semaphoreImageAvailable;
  std::unique_ptr<Semaphore> _semaphoreInFlight;
  uint64_t _valueSemaphoreInFlight = 1;
  int _maxFramesInFlight;
  int _frameInFlight = 0;

  struct Cache {
    bool queueTypeChange = false;
    GraphPass* previousPass = nullptr;
  };

  std::map<GraphPass*, Cache> _cache;

 public:
  Graph(int threadsNumber, int maxFramesInFlight, Swapchain& swapchain, const Window& window, const Device& device) noexcept;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  Graph(Graph&&) = delete;
  Graph& operator=(Graph&&) = delete;

  void initialize() noexcept;
  GraphPassGraphic& createPassGraphic(std::string_view name);
  GraphPassCompute& createPassCompute(std::string_view name, bool separate);
  GraphPassGraphic* getPassGraphic(std::string_view name) const noexcept;
  GraphPassCompute* getPassCompute(std::string_view name) const noexcept;
  GraphStorage& getGraphStorage() const noexcept;
  std::map<std::string, glm::dvec2> getTimestamps() const noexcept;
  int getFrameInFlight() const noexcept;

  void calculate();
  // true -> need to call reset
  bool render();
  void reset();

  void print() const noexcept;
};
}  // namespace RenderGraph