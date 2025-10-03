export module Instance;
import <volk.h>;
import <VkBootstrap.h>;
import <iostream>;
import <string>;

export namespace RenderGraph {
class Instance final {
 private:
  vkb::Instance _instance;
  bool _debugUtils = false;

 public:
  Instance(std::string_view name, bool validation);
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;
  Instance(Instance&&) = delete;
  Instance& operator=(Instance&&) = delete;

  bool isDebug() const noexcept;
  const vkb::Instance& getInstance() const noexcept;
  ~Instance();
};
}  // namespace RenderGraph