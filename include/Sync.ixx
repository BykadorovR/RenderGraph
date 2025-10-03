export module Sync;
import Device;
import <volk.h>;

export namespace RenderGraph {
class Semaphore {
 private:
  const Device* _device;
  VkSemaphore _semaphore;
  VkSemaphoreType _type;

 public:
  Semaphore(VkSemaphoreType type, const Device& device);
  Semaphore(const Semaphore&) = delete;
  Semaphore& operator=(const Semaphore&) = delete;
  Semaphore(Semaphore&&) = delete;
  Semaphore& operator=(Semaphore&&) = delete;

  VkSemaphore getSemaphore() const noexcept;
  ~Semaphore();
};
}  // namespace RenderGraph