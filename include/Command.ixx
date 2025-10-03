export module Command;
import Device;
import CommandPool;
import <volk.h>;

export namespace RenderGraph {
class CommandBuffer final {
 private:
  const CommandPool* _pool;
  const Device* _device;
  VkCommandBuffer _buffer;
  bool _active = false;

 public:
  CommandBuffer(const CommandPool& pool, const Device& device);
  CommandBuffer(CommandBuffer&& commandBuffer) noexcept;
  CommandBuffer(const CommandBuffer&) = delete;
  CommandBuffer& operator=(const CommandBuffer&) = delete;
  CommandBuffer& operator=(CommandBuffer&&) = delete;

  void beginCommands() noexcept;
  void endCommands() noexcept;
  bool getActive() const noexcept;
  const VkCommandBuffer& getCommandBuffer() const noexcept;
  ~CommandBuffer();
};
}  // namespace RenderGraph