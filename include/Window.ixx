module;

#if defined(__ANDROID__)
#include <android/native_window.h>
#else
#include <GLFW/glfw3.h>
#endif

export module Window;

import glm;

export namespace RenderGraph {
class Window final {
 private:
#if defined(__ANDROID__)
  ANativeWindow* _window = nullptr;
#else
  GLFWwindow* _window = nullptr;
#endif
  glm::ivec2 _resolution;
  bool _resized = false;

 public:
#if defined(__ANDROID__)
  // Retains its own reference to the native window for the lifetime of this object.
  explicit Window(ANativeWindow* window) noexcept;
#else
  Window(glm::ivec2 resolution) noexcept;
#endif
  glm::ivec2 getResolution() const noexcept;
  void initialize();
#if defined(__ANDROID__)
  ANativeWindow* getWindow() const noexcept;
#else
  // GLFW API expects a non-const window pointer.
  GLFWwindow* getWindow() const noexcept;
#endif
  // Android window initialization and fullscreen state are managed by the host Activity.
  void setFullScreen(bool fullScreen);
  ~Window();
};
}  // namespace RenderGraph
