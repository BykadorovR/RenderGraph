export module Window;
import glm;
import <GLFW/glfw3.h>;

export namespace RenderGraph {
class Window final {
 private:
  GLFWwindow* _window = nullptr;
  glm::ivec2 _resolution;
  bool _resized = false;

 public:
  Window(glm::ivec2 resolution) noexcept;
  glm::ivec2 getResolution() const noexcept;
  void initialize();
  // glfw API expects non-const window pointer
  GLFWwindow* getWindow() const noexcept;
  void setFullScreen(bool fullScreen);
  ~Window();
};
}  // namespace RenderGraph