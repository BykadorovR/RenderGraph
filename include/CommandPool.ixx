export module CommandPool;
import Device;
import <volk.h>;
import <VkBootstrap.h>;
import <map>;

export namespace RenderGraph {
class CommandPool final {
 private:
  const Device* _device;
  vkb::QueueType _type;
  VkCommandPool _commandPool;

 public:
  CommandPool(vkb::QueueType type, const Device& device);
  CommandPool(const CommandPool&) = delete;
  CommandPool& operator=(const CommandPool&) = delete;
  CommandPool(CommandPool&&) = delete;
  CommandPool& operator=(CommandPool&&) = delete;

  VkCommandPool getCommandPool() const noexcept;
  vkb::QueueType getType() const noexcept;
  ~CommandPool();
};
}  // namespace RenderGraph