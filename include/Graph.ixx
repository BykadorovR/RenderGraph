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
import <unordered_set>;

// Forward declarations for test classes, not visible outside this module
class ScenarioTest_GraphSeparateQueues_Test;
class ScenarioTest_BufferOwnershipTransferUsesLastResourceOwner_Test;
class ScenarioTest_GraphOneQueue_Test;
class ScenarioTest_TraversalKeepsTransitiveProducerBeforeConsumer_Test;
class ScenarioTest_TraversalDiamondGraph_Test;

export namespace RenderGraph {
class GraphStorage final {
 private:
  std::unordered_map<std::string, std::unique_ptr<ImageViewHolder>> _imageViewHolders;
  std::unordered_map<std::string, std::vector<std::unique_ptr<Buffer>>> _buffers;

 public:
  GraphStorage() = default;
  GraphStorage(const GraphStorage&) = delete;
  GraphStorage& operator=(const GraphStorage&) = delete;
  GraphStorage(GraphStorage&&) = delete;
  GraphStorage& operator=(GraphStorage&&) = delete;

  void add(std::string_view name, std::unique_ptr<ImageViewHolder> imageViewHolder) noexcept;
  // not const because will do std::move
  void add(std::string_view name, std::vector<std::unique_ptr<Buffer>>& buffers) noexcept;
  void reset(std::vector<std::shared_ptr<ImageView>> oldSwapchain,
             std::vector<std::shared_ptr<ImageView>> newSwapchain) noexcept;
  std::string find(const std::vector<std::shared_ptr<ImageView>>& imageViews) noexcept;
  bool containsImageViewHolder(std::string_view name) const noexcept;
  const ImageViewHolder& getImageViewHolder(std::string_view name) const;
  bool containsBuffer(std::string_view name) const noexcept;
  // NVRO
  std::vector<Buffer*> getBuffer(std::string_view name) const;
};

class GraphElement {
 public:
  virtual void draw(int currentFrame, const CommandBuffer& commandBuffer) = 0;
  virtual void update(int currentFrame, const CommandBuffer& commandBuffer) = 0;
  virtual void reset(const std::vector<std::shared_ptr<RenderGraph::ImageView>>& swapchain) = 0;
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
  std::vector<std::shared_ptr<GraphElement>> _graphElements;

 public:
  GraphPass(std::string_view name, GraphPassType graphPassType, const GraphStorage& graphStorage) noexcept;
  GraphPass(const GraphPass&) = delete;
  GraphPass& operator=(const GraphPass&) = delete;
  GraphPass(GraphPass&&) = delete;
  GraphPass& operator=(GraphPass&&) = delete;

  void registerGraphElement(std::shared_ptr<GraphElement> graphElement) noexcept;
  // NVRO
  GraphPassType getGraphPassType() const noexcept;
  std::vector<CommandBuffer*> getCommandBuffers() const noexcept;
  std::string getName() const noexcept;
  virtual void execute(int currentFrame, const CommandBuffer& commandBuffer) = 0;
  void reset(const std::vector<std::shared_ptr<RenderGraph::ImageView>>& swapchain);
  virtual ~GraphPass() = default;
};

class GraphPassGraphic final : public GraphPass {
 private:
  std::vector<std::string> _colorTargets, _textureInputs;
  std::optional<std::string> _depthTarget;

  std::unordered_map<std::string, bool> _clearTarget;
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
  friend class ::ScenarioTest_GraphSeparateQueues_Test;
  friend class ::ScenarioTest_BufferOwnershipTransferUsesLastResourceOwner_Test;
  friend class ::ScenarioTest_GraphOneQueue_Test;
  friend class ::ScenarioTest_TraversalKeepsTransitiveProducerBeforeConsumer_Test;
  friend class ::ScenarioTest_TraversalDiamondGraph_Test;

  Swapchain* _swapchain;
  const Device* _device;
  const Window* _window;
  std::unique_ptr<BS::thread_pool> _threadPool;
  std::vector<std::unique_ptr<GraphPass>> _passes;
  std::deque<GraphPass*> _passesOrdered;
  std::unique_ptr<Timestamps> _timestamps;
  std::unique_ptr<GraphStorage> _graphStorage;
  // special semaphores
  std::vector<std::shared_ptr<Semaphore>> _semaphoreRenderFinished, _semaphoreImageAvailable;
  std::unique_ptr<Semaphore> _semaphoreInFlight;
  uint64_t _valueSemaphoreInFlight = 1;
  int _maxFramesInFlight;
  int _frameInFlight = 0;

  // resources per pass
  struct Resource {
    std::string name;
    enum class Type { IMAGE, BUFFER } type;
    enum class Operation : uint8_t {
      READ = 1 << 0,
      WRITE = 1 << 1,
    };
    uint8_t operation = 0;
  };

  class Resources {
   private:
    std::vector<Resource> _resources;

   public:
    void add(Resource resource);
    bool contains(std::string_view name, Resource::Type type, Resource::Operation operation) const;
    const std::vector<Resource>& getResources() const noexcept;
    std::vector<Resource> getResources(Resource::Operation operation) const;
    std::vector<std::string> getNames(Resource::Type type) const;
  };

  std::unordered_map<GraphPass*, Resources> _resources;

  // syncrhonization per pass
  class Sync {
   private:
    // same semaphore is wait and signal, so can't be unique
    std::vector<std::pair<std::vector<std::shared_ptr<Semaphore>>, std::function<int()>>> _waitSemaphores,
        _signalSemaphores;
    std::vector<std::pair<std::vector<Barrier>, std::function<int()>>> _barriersBefore, _barriersAfter;

   public:
    Sync() = default;
    void addSignalSemaphore(std::vector<std::shared_ptr<Semaphore>>& signalSemaphore,
                            std::function<int()> index) noexcept;
    void addWaitSemaphore(std::vector<std::shared_ptr<Semaphore>>& waitSemaphore, std::function<int()> index) noexcept;
    void addBarrierBefore(std::vector<Barrier>& barriers, std::function<int()> index) noexcept;
    void addBarrierAfter(std::vector<Barrier>& barriers, std::function<int()> index) noexcept;

    std::vector<Semaphore*> getWaitSemaphores() const noexcept;
    std::vector<Semaphore*> getSignalSemaphores() const noexcept;
    std::vector<const Barrier*> getBarriersBefore() const noexcept;
    std::vector<const Barrier*> getBarriersAfter() const noexcept;
  };
  std::unordered_map<GraphPass*, Sync> _sync;

  bool _usesSeparateQueue(GraphPass* pass);

 public:
  Graph(int threadsNumber,
        int maxFramesInFlight,
        Swapchain& swapchain,
        const Window& window,
        const Device& device) noexcept;
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
  std::unordered_map<std::string, glm::dvec2> getTimestamps() const noexcept;
  int getFrameInFlight() const noexcept;

  void calculate();
  // true -> need to call reset
  bool render();
  void reset();

  void print() const noexcept;
};
}  // namespace RenderGraph