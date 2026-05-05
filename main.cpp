/*
 * Custom Cross-Platform Game Engine for Android
 * Features: OpenGL ES 3.0+, ECS Architecture, PBR Lighting, Multi-touch Input
 * Compile with CMake and Android NDK
 */

#include <android_native_app_glue.h>
#include <android/log.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <chrono>
#include <algorithm>

#define LOG_TAG "GameEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ============================================================================
// MATHEMATICS LIBRARY
// ============================================================================
struct Vec2 {
    float x, y;
    Vec2() : x(0), y(0) {}
    Vec2(float x, float y) : x(x), y(y) {}
    Vec2 operator+(const Vec2& other) const { return Vec2(x + other.x, y + other.y); }
    Vec2 operator-(const Vec2& other) const { return Vec2(x - other.x, y - other.y); }
    Vec2 operator*(float scalar) const { return Vec2(x * scalar, y * scalar); }
    float length() const { return sqrtf(x * x + y * y); }
    Vec2 normalized() const { float len = length(); return len > 0 ? Vec2(x/len, y/len) : Vec2(0,0); }
};

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& other) const { return Vec3(x + other.x, y + other.y, z + other.z); }
    Vec3 operator-(const Vec3& other) const { return Vec3(x - other.x, y - other.y, z - other.z); }
    Vec3 operator*(float scalar) const { return Vec3(x * scalar, y * scalar, z * scalar); }
    Vec3 cross(const Vec3& other) const {
        return Vec3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }
    float dot(const Vec3& other) const { return x * other.x + y * other.y + z * other.z; }
    float length() const { return sqrtf(x * x + y * y + z * z); }
    Vec3 normalized() const { float len = length(); return len > 0 ? Vec3(x/len, y/len, z/len) : Vec3(0,0,0); }
};

struct Vec4 {
    float x, y, z, w;
    Vec4() : x(0), y(0), z(0), w(1) {}
    Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
};

struct Mat4 {
    float m[16];
    
    Mat4() { setIdentity(); }
    
    void setIdentity() {
        for(int i = 0; i < 16; i++) m[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    }
    
    static Mat4 perspective(float fovY, float aspect, float near, float far) {
        Mat4 result;
        float f = 1.0f / tanf(fovY * 0.5f);
        float rangeInv = 1.0f / (near - far);
        
        result.m[0] = f / aspect; result.m[1] = 0; result.m[2] = 0; result.m[3] = 0;
        result.m[4] = 0; result.m[5] = f; result.m[6] = 0; result.m[7] = 0;
        result.m[8] = 0; result.m[9] = 0; result.m[10] = (near + far) * rangeInv; result.m[11] = -1;
        result.m[12] = 0; result.m[13] = 0; result.m[14] = near * far * rangeInv * 2; result.m[15] = 0;
        
        return result;
    }
    
    static Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
        Mat4 result;
        Vec3 f = (center - eye).normalized();
        Vec3 s = f.cross(up).normalized();
        Vec3 u = s.cross(f);
        
        result.m[0] = s.x; result.m[4] = s.y; result.m[8] = s.z; result.m[12] = -s.dot(eye);
        result.m[1] = u.x; result.m[5] = u.y; result.m[9] = u.z; result.m[13] = -u.dot(eye);
        result.m[2] = -f.x; result.m[6] = -f.y; result.m[10] = -f.z; result.m[14] = f.dot(eye);
        result.m[3] = 0; result.m[7] = 0; result.m[11] = 0; result.m[15] = 1;
        
        return result;
    }
    
    Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < 4; j++) {
                result.m[j * 4 + i] = m[i] * other.m[j * 4] + m[4 + i] * other.m[j * 4 + 1] + 
                                     m[8 + i] * other.m[j * 4 + 2] + m[12 + i] * other.m[j * 4 + 3];
            }
        }
        return result;
    }
    
    static Mat4 rotation(float angle, const Vec3& axis) {
        Mat4 result;
        Vec3 a = axis.normalized();
        float c = cosf(angle), s = sinf(angle);
        float t = 1.0f - c;
        
        result.m[0] = t * a.x * a.x + c; result.m[4] = t * a.x * a.y - s * a.z; result.m[8] = t * a.x * a.z + s * a.y; result.m[12] = 0;
        result.m[1] = t * a.x * a.y + s * a.z; result.m[5] = t * a.y * a.y + c; result.m[9] = t * a.y * a.z - s * a.x; result.m[13] = 0;
        result.m[2] = t * a.x * a.z - s * a.y; result.m[6] = t * a.y * a.z + s * a.x; result.m[10] = t * a.z * a.z + c; result.m[14] = 0;
        result.m[3] = 0; result.m[7] = 0; result.m[11] = 0; result.m[15] = 1;
        
        return result;
    }
    
    static Mat4 translation(const Vec3& pos) {
        Mat4 result;
        result.m[12] = pos.x; result.m[13] = pos.y; result.m[14] = pos.z;
        return result;
    }
    
    static Mat4 scale(const Vec3& s) {
        Mat4 result;
        result.m[0] = s.x; result.m[5] = s.y; result.m[10] = s.z;
        return result;
    }
};

// ============================================================================
// ENTITY COMPONENT SYSTEM (ECS)
// ============================================================================
typedef uint32_t Entity;
typedef uint32_t ComponentType;
const Entity MAX_ENTITIES = 1024;
const ComponentType MAX_COMPONENTS = 32;

class ComponentBase {
public:
    virtual ~ComponentBase() = default;
};

// Transform Component
struct Transform : ComponentBase {
    Vec3 position;
    Vec3 rotation;
    Vec3 scale;
    Transform() : position(0,0,0), rotation(0,0,0), scale(1,1,1) {}
};

// Mesh Component
struct MeshComponent : ComponentBase {
    GLuint vao, vbo, ebo;
    int indexCount;
    GLenum primitiveType;
    Vec3 color;
    float metallic;
    float roughness;
};

// Camera Component
struct CameraComponent : ComponentBase {
    float fov;
    float nearPlane, farPlane;
    Vec3 target;
    CameraComponent() : fov(45.0f), nearPlane(0.1f), farPlane(100.0f), target(0,0,0) {}
};

// Light Component
struct LightComponent : ComponentBase {
    Vec3 color;
    float intensity;
    bool isDirectional;
    LightComponent() : color(1,1,1), intensity(1.0f), isDirectional(false) {}
};

class ECSWorld {
private:
    struct ComponentPool {
        void* data;
        size_t elementSize;
        std::vector<bool> entityMap;
        
        ComponentPool(size_t size) : elementSize(size) {
            data = malloc(size * MAX_ENTITIES);
            entityMap.resize(MAX_ENTITIES, false);
        }
        ~ComponentPool() { free(data); }
    };
    
    std::unordered_map<ComponentType, std::unique_ptr<ComponentPool>> componentPools;
    std::vector<bool> activeEntities;
    
public:
    ECSWorld() : activeEntities(MAX_ENTITIES, false) {}
    
    Entity createEntity() {
        for(Entity e = 0; e < MAX_ENTITIES; ++e) {
            if(!activeEntities[e]) {
                activeEntities[e] = true;
                return e;
            }
        }
        return MAX_ENTITIES;
    }
    
    template<typename T>
    void registerComponent() {
        ComponentType type = getComponentType<T>();
        if(componentPools.find(type) == componentPools.end()) {
            componentPools[type] = std::make_unique<ComponentPool>(sizeof(T));
        }
    }
    
    template<typename T>
    T* addComponent(Entity entity) {
        ComponentType type = getComponentType<T>();
        auto& pool = componentPools[type];
        void* ptr = static_cast<char*>(pool->data) + entity * sizeof(T);
        pool->entityMap[entity] = true;
        return new(ptr) T();
    }
    
    template<typename T>
    T* getComponent(Entity entity) {
        ComponentType type = getComponentType<T>();
        auto& pool = componentPools[type];
        if(pool->entityMap[entity])
            return static_cast<T*>(static_cast<char*>(pool->data) + entity * sizeof(T));
        return nullptr;
    }
    
private:
    template<typename T>
    static ComponentType getComponentType() {
        static ComponentType type = nextComponentType++;
        return type;
    }
    static ComponentType nextComponentType;
};
ComponentType ECSWorld::nextComponentType = 0;

// ============================================================================
// SHADER MANAGER
// ============================================================================
class ShaderManager {
private:
    GLuint pbrShader;
    GLuint uiShader;
    
    const char* pbrVertexShader = R"(#version 300 es
        precision highp float;
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aNormal;
        layout(location = 2) in vec2 aTexCoord;
        
        uniform mat4 uModel;
        uniform mat4 uView;
        uniform mat4 uProjection;
        uniform mat3 uNormalMatrix;
        
        out vec3 vWorldPos;
        out vec3 vNormal;
        out vec2 vTexCoord;
        
        void main() {
            vec4 worldPos = uModel * vec4(aPos, 1.0);
            vWorldPos = worldPos.xyz;
            vNormal = normalize(uNormalMatrix * aNormal);
            vTexCoord = aTexCoord;
            gl_Position = uProjection * uView * worldPos;
        }
    )";
    
    const char* pbrFragmentShader = R"(#version 300 es
        precision highp float;
        
        in vec3 vWorldPos;
        in vec3 vNormal;
        in vec2 vTexCoord;
        
        uniform vec3 uLightPos;
        uniform vec3 uLightColor;
        uniform float uLightIntensity;
        uniform vec3 uCameraPos;
        uniform vec3 uAlbedo;
        uniform float uMetallic;
        uniform float uRoughness;
        uniform float uAmbientStrength;
        
        out vec4 FragColor;
        
        const float PI = 3.14159265359;
        
        float DistributionGGX(vec3 N, vec3 H, float roughness) {
            float a = roughness * roughness;
            float a2 = a * a;
            float NdotH = max(dot(N, H), 0.0);
            float NdotH2 = NdotH * NdotH;
            float nom = a2;
            float denom = (NdotH2 * (a2 - 1.0) + 1.0);
            return nom / (PI * denom * denom);
        }
        
        float GeometrySchlickGGX(float NdotV, float roughness) {
            float r = roughness + 1.0;
            float k = (r * r) / 8.0;
            float nom = NdotV;
            float denom = NdotV * (1.0 - k) + k;
            return nom / denom;
        }
        
        float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
            float NdotV = max(dot(N, V), 0.0);
            float NdotL = max(dot(N, L), 0.0);
            return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
        }
        
        vec3 FresnelSchlick(float cosTheta, vec3 F0) {
            return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
        }
        
        void main() {
            vec3 N = normalize(vNormal);
            vec3 V = normalize(uCameraPos - vWorldPos);
            vec3 L = normalize(uLightPos - vWorldPos);
            vec3 H = normalize(V + L);
            
            vec3 F0 = mix(vec3(0.04), uAlbedo, uMetallic);
            
            // Cook-Torrance BRDF
            float NDF = DistributionGGX(N, H, uRoughness);
            float G = GeometrySmith(N, V, L, uRoughness);
            vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
            
            vec3 kS = F;
            vec3 kD = vec3(1.0) - kS;
            kD *= 1.0 - uMetallic;
            
            float NdotL = max(dot(N, L), 0.0);
            vec3 nominator = NDF * G * F;
            float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.001;
            vec3 specular = nominator / denominator;
            
            // Radiance
            float distance = length(uLightPos - vWorldPos);
            float attenuation = 1.0 / (distance * distance);
            vec3 radiance = uLightColor * uLightIntensity * attenuation;
            
            vec3 ambient = vec3(uAmbientStrength) * uAlbedo;
            vec3 color = ambient + (kD * uAlbedo / PI + specular) * radiance * NdotL;
            
            // HDR Tone Mapping
            color = color / (color + vec3(1.0));
            // Gamma Correction
            color = pow(color, vec3(1.0/2.2));
            
            FragColor = vec4(color, 1.0);
        }
    )";
    
    GLuint compileShader(GLenum type, const char* source) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, NULL);
        glCompileShader(shader);
        
        GLint compiled;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if(!compiled) {
            GLchar infoLog[512];
            glGetShaderInfoLog(shader, 512, NULL, infoLog);
            LOGE("Shader compilation failed: %s", infoLog);
        }
        return shader;
    }
    
    GLuint createProgram(const char* vertexSource, const char* fragmentSource) {
        GLuint vertex = compileShader(GL_VERTEX_SHADER, vertexSource);
        GLuint fragment = compileShader(GL_FRAGMENT_SHADER, fragmentSource);
        
        GLuint program = glCreateProgram();
        glAttachShader(program, vertex);
        glAttachShader(program, fragment);
        glLinkProgram(program);
        
        GLint linked;
        glGetProgramiv(program, GL_LINK_STATUS, &linked);
        if(!linked) {
            GLchar infoLog[512];
            glGetProgramInfoLog(program, 512, NULL, infoLog);
            LOGE("Program linking failed: %s", infoLog);
        }
        
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        
        return program;
    }
    
public:
    void initialize() {
        pbrShader = createProgram(pbrVertexShader, pbrFragmentShader);
        LOGI("PBR Shader created with ID: %d", pbrShader);
    }
    
    GLuint getPBRShader() { return pbrShader; }
    
    void cleanup() {
        if(pbrShader) glDeleteProgram(pbrShader);
        if(uiShader) glDeleteProgram(uiShader);
    }
};

// ============================================================================
// INPUT SYSTEM
// ============================================================================
enum class TouchEventType { DOWN, MOVE, UP };

struct TouchEvent {
    TouchEventType type;
    float x, y;
    int pointerId;
};

class InputSystem {
private:
    std::vector<TouchEvent> touchEvents;
    Vec2 joystickCenter;
    float joystickRadius;
    Vec2 joystickPosition;
    bool joystickActive;
    
    Vec2 lastTouchPosition;
    Vec2 cameraRotation;
    bool cameraDragging;
    
public:
    InputSystem() : joystickRadius(100.0f), joystickActive(false), 
                   cameraDragging(false), cameraRotation(0,0), 
                   joystickPosition(0,0), joystickCenter(0,0) {}
    
    void processInput(AInputEvent* event) {
        int32_t type = AInputEvent_getType(event);
        
        if(type == AINPUT_EVENT_TYPE_MOTION) {
            int32_t action = AMotionEvent_getAction(event);
            int32_t pointerIndex = (action & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
            int32_t pointerId = AMotionEvent_getPointerId(event, pointerIndex);
            
            float x = AMotionEvent_getX(event, pointerIndex);
            float y = AMotionEvent_getY(event, pointerIndex);
            
            TouchEvent touchEvent;
            touchEvent.x = x;
            touchEvent.y = y;
            touchEvent.pointerId = pointerId;
            
            switch(action & AMOTION_EVENT_ACTION_MASK) {
                case AMOTION_EVENT_ACTION_DOWN:
                case AMOTION_EVENT_ACTION_POINTER_DOWN:
                    touchEvent.type = TouchEventType::DOWN;
                    if(x < 400.0f) {  // Left side = joystick
                        joystickCenter = Vec2(x, y);
                        joystickPosition = Vec2(0, 0);
                        joystickActive = true;
                    } else {  // Right side = camera
                        lastTouchPosition = Vec2(x, y);
                        cameraDragging = true;
                    }
                    break;
                    
                case AMOTION_EVENT_ACTION_MOVE:
                    touchEvent.type = TouchEventType::MOVE;
                    if(joystickActive && x < 400.0f) {
                        Vec2 diff = Vec2(x, y) - joystickCenter;
                        float distance = diff.length();
                        if(distance > joystickRadius) {
                            diff = diff.normalized() * joystickRadius;
                        }
                        joystickPosition = diff * (1.0f / joystickRadius);
                    }
                    if(cameraDragging && x >= 400.0f) {
                        Vec2 currentPos(x, y);
                        Vec2 delta = currentPos - lastTouchPosition;
                        cameraRotation = cameraRotation + delta * 0.005f;
                        cameraRotation.y = std::max(-1.5f, std::min(1.5f, cameraRotation.y));
                        lastTouchPosition = currentPos;
                    }
                    break;
                    
                case AMOTION_EVENT_ACTION_UP:
                case AMOTION_EVENT_ACTION_POINTER_UP:
                    touchEvent.type = TouchEventType::UP;
                    joystickActive = false;
                    joystickPosition = Vec2(0, 0);
                    cameraDragging = false;
                    break;
            }
            
            touchEvents.push_back(touchEvent);
        }
    }
    
    Vec2 getJoystickInput() { return joystickPosition; }
    Vec2 getCameraRotation() { return cameraRotation; }
    bool isJoystickActive() { return joystickActive; }
    void clearEvents() { touchEvents.clear(); }
};

// ============================================================================
// RENDERER
// ============================================================================
class Renderer {
private:
    ShaderManager* shaderManager;
    GLuint cubeVAO, cubeVBO, cubeEBO;
    int cubeIndexCount;
    
    void createCube() {
        // PBR Cube with normals and texture coordinates
        float vertices[] = {
            // Position          // Normal           // TexCoord     // Color
            // Front face
            -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // Red-ish
             0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
            -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
            
            // Back face
            -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,  // Green-ish
             0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
             0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            
            // Top face
            -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // Blue-ish
             0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
             0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
            -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
            
            // Bottom face
            -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,  // Yellow-ish
             0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
             0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
            -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f,
            
            // Right face
             0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,  // Purple-ish
             0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
             0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
             0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f,
            
            // Left face
            -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,  // Cyan-ish
            -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
            -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
            -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f
        };
        
        unsigned int indices[] = {
            0, 1, 2, 2, 3, 0,       // Front
            4, 5, 6, 6, 7, 4,       // Back
            8, 9, 10, 10, 11, 8,    // Top
            12, 13, 14, 14, 15, 12, // Bottom
            16, 17, 18, 18, 19, 16, // Right
            20, 21, 22, 22, 23, 20  // Left
        };
        
        cubeIndexCount = 36;
        
        glGenVertexArrays(1, &cubeVAO);
        glGenBuffers(1, &cubeVBO);
        glGenBuffers(1, &cubeEBO);
        
        glBindVertexArray(cubeVAO);
        
        glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        
        // Position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // Normal
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        // TexCoord
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        
        glBindVertexArray(0);
    }
    
public:
    void initialize(ShaderManager* sm) {
        shaderManager = sm;
        createCube();
        
        // Enable depth testing and face culling
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        
        LOGI("Renderer initialized with PBR cube");
    }
    
    void renderCube(const Mat4& model, const Mat4& view, const Mat4& projection, 
                   const Vec3& cameraPos, const Vec3& lightPos, float time) {
        glUseProgram(shaderManager->getPBRShader());
        
        // Set matrices
        GLuint modelLoc = glGetUniformLocation(shaderManager->getPBRShader(), "uModel");
        GLuint viewLoc = glGetUniformLocation(shaderManager->getPBRShader(), "uView");
        GLuint projLoc = glGetUniformLocation(shaderManager->getPBRShader(), "uProjection");
        
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, model.m);
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view.m);
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection.m);
        
        // Calculate normal matrix
        Mat4 normalMat4 = model;
        float normalMat3[9] = {
            normalMat4.m[0], normalMat4.m[1], normalMat4.m[2],
            normalMat4.m[4], normalMat4.m[5], normalMat4.m[6],
            normalMat4.m[8], normalMat4.m[9], normalMat4.m[10]
        };
        GLuint normalMatLoc = glGetUniformLocation(shaderManager->getPBRShader(), "uNormalMatrix");
        glUniformMatrix3fv(normalMatLoc, 1, GL_FALSE, normalMat3);
        
        // PBR parameters
        glUniform3f(glGetUniformLocation(shaderManager->getPBRShader(), "uLightPos"), 
                   lightPos.x, lightPos.y, lightPos.z);
        glUniform3f(glGetUniformLocation(shaderManager->getPBRShader(), "uLightColor"), 
                   1.0f, 1.0f, 0.9f);
        glUniform1f(glGetUniformLocation(shaderManager->getPBRShader(), "uLightIntensity"), 5.0f);
        glUniform3f(glGetUniformLocation(shaderManager->getPBRShader(), "uCameraPos"), 
                   cameraPos.x, cameraPos.y, cameraPos.z);
        glUniform3f(glGetUniformLocation(shaderManager->getPBRShader(), "uAlbedo"), 
                   0.8f + sinf(time * 2.0f) * 0.2f, 0.5f, 0.7f);
        glUniform1f(glGetUniformLocation(shaderManager->getPBRShader(), "uMetallic"), 0.3f);
        glUniform1f(glGetUniformLocation(shaderManager->getPBRShader(), "uRoughness"), 0.4f);
        glUniform1f(glGetUniformLocation(shaderManager->getPBRShader(), "uAmbientStrength"), 0.1f);
        
        glBindVertexArray(cubeVAO);
        glDrawElements(GL_TRIANGLES, cubeIndexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
    
    void clear() {
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    
    void cleanup() {
        glDeleteVertexArrays(1, &cubeVAO);
        glDeleteBuffers(1, &cubeVBO);
        glDeleteBuffers(1, &cubeEBO);
    }
};

// ============================================================================
// PHYSICS ENGINE (Simplified)
// ============================================================================
class PhysicsEngine {
private:
    struct PhysicsBody {
        Vec3 position;
        Vec3 velocity;
        Vec3 acceleration;
        float mass;
        bool isStatic;
        float radius; // For simple sphere collision
    };
    
    std::vector<PhysicsBody> bodies;
    Vec3 gravity;
    
public:
    PhysicsEngine() : gravity(0, -9.81f, 0) {}
    
    int addBody(const Vec3& pos, float mass = 1.0f, bool isStatic = false, float radius = 0.5f) {
        PhysicsBody body = {pos, Vec3(0,0,0), Vec3(0,0,0), mass, isStatic, radius};
        bodies.push_back(body);
        return bodies.size() - 1;
    }
    
    void update(float deltaTime) {
        for(auto& body : bodies) {
            if(body.isStatic) continue;
            
            body.acceleration = gravity;
            body.velocity = body.velocity + body.acceleration * deltaTime;
            body.position = body.position + body.velocity * deltaTime;
            
            // Simple ground collision
            if(body.position.y - body.radius < 0.0f) {
                body.position.y = body.radius;
                body.velocity.y *= -0.5f; // Bounce with damping
            }
        }
        
        // Simple sphere-sphere collision
        for(size_t i = 0; i < bodies.size(); ++i) {
            for(size_t j = i + 1; j < bodies.size(); ++j) {
                Vec3 diff = bodies[i].position - bodies[j].position;
                float distance = diff.length();
                float minDist = bodies[i].radius + bodies[j].radius;
                
                if(distance < minDist) {
                    Vec3 normal = diff.normalized();
                    Vec3 relativeVelocity = bodies[i].velocity - bodies[j].velocity;
                    float velAlongNormal = relativeVelocity.dot(normal);
                    
                    if(velAlongNormal > 0) continue;
                    
                    float restitution = 0.5f;
                    float impulseMagnitude = -(1.0f + restitution) * velAlongNormal;
                    impulseMagnitude /= (1.0f / bodies[i].mass + 1.0f / bodies[j].mass);
                    
                    Vec3 impulse = normal * impulseMagnitude;
                    
                    if(!bodies[i].isStatic) bodies[i].velocity = bodies[i].velocity + impulse * (1.0f / bodies[i].mass);
                    if(!bodies[j].isStatic) bodies[j].velocity = bodies[j].velocity - impulse * (1.0f / bodies[j].mass);
                    
                    // Separate bodies
                    float overlap = minDist - distance;
                    Vec3 separation = normal * overlap * 0.5f;
                    if(!bodies[i].isStatic) bodies[i].position = bodies[i].position + separation;
                    if(!bodies[j].isStatic) bodies[j].position = bodies[j].position - separation;
                }
            }
        }
    }
    
    Vec3 getPosition(int index) { return bodies[index].position; }
    void setPosition(int index, const Vec3& pos) { bodies[index].position = pos; }
    void setVelocity(int index, const Vec3& vel) { bodies[index].velocity = vel; }
};

// ============================================================================
// UI SYSTEM
// ============================================================================
class UISystem {
private:
    struct UIElement {
        Vec2 position;
        Vec2 size;
        Vec4 color;
        std::string text;
        bool isButton;
        std::function<void()> onClick;
    };
    
    std::vector<UIElement> elements;
    
public:
    void addButton(float x, float y, float width, float height, Vec4 color, std::function<void()> callback) {
        UIElement element;
        element.position = Vec2(x, y);
        element.size = Vec2(width, height);
        element.color = color;
        element.isButton = true;
        element.onClick = callback;
        elements.push_back(element);
    }
    
    void addText(float x, float y, const std::string& text, Vec4 color = Vec4(1,1,1,1)) {
        UIElement element;
        element.position = Vec2(x, y);
        element.text = text;
        element.color = color;
        element.isButton = false;
        elements.push_back(element);
    }
    
    void render2DOverlay(int screenWidth, int screenHeight) {
        // Save 3D state
        GLboolean depthTest = glIsEnabled(GL_DEPTH_TEST);
        GLboolean cullFace = glIsEnabled(GL_CULL_FACE);
        
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        
        // Simple 2D rendering using immediate mode substitute
        for(const auto& element : elements) {
            if(element.isButton) {
                // Render button as colored quad
                // In production, use proper 2D shader
                GLfloat vertices[] = {
                    element.position.x, element.position.y,
                    element.position.x + element.size.x, element.position.y,
                    element.position.x + element.size.x, element.position.y + element.size.y,
                    element.position.x, element.position.y + element.size.y
                };
            }
        }
        
        // Restore 3D state
        if(depthTest) glEnable(GL_DEPTH_TEST);
        if(cullFace) glEnable(GL_CULL_FACE);
    }
};

// ============================================================================
// MAIN ENGINE CLASS
// ============================================================================
class GameEngine {
private:
    android_app* app;
    
    // EGL
    EGLDisplay display;
    EGLSurface surface;
    EGLContext context;
    int width, height;
    
    // Core systems
    std::unique_ptr<ShaderManager> shaderManager;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<ECSWorld> ecsWorld;
    std::unique_ptr<InputSystem> inputSystem;
    std::unique_ptr<PhysicsEngine> physicsEngine;
    std::unique_ptr<UISystem> uiSystem;
    
    // Timing
    std::chrono::high_resolution_clock::time_point lastFrameTime;
    float totalTime;
    
    // Camera
    Vec3 cameraPos;
    Vec3 cameraTarget;
    float cameraDistance;
    
    // Physics object index
    int cubePhysicsIndex;
    
public:
    GameEngine(android_app* app) : app(app), totalTime(0), cameraDistance(5.0f) {
        app->userData = this;
        app->onAppCmd = handleAppCommand;
        app->onInputEvent = handleInputEvent;
    }
    
    static void handleAppCommand(android_app* app, int32_t cmd) {
        GameEngine* engine = static_cast<GameEngine*>(app->userData);
        switch(cmd) {
            case APP_CMD_INIT_WINDOW:
                LOGI("Window initialized");
                if(app->window != nullptr) {
                    engine->initialize();
                }
                break;
            case APP_CMD_TERM_WINDOW:
                LOGI("Window terminated");
                engine->cleanup();
                break;
        }
    }
    
    static int32_t handleInputEvent(android_app* app, AInputEvent* event) {
        GameEngine* engine = static_cast<GameEngine*>(app->userData);
        engine->inputSystem->processInput(event);
        return 1;
    }
    
    void initialize() {
        // Initialize EGL
        initializeEGL();
        
        // Initialize systems
        shaderManager = std::make_unique<ShaderManager>();
        shaderManager->initialize();
        
        renderer = std::make_unique<Renderer>();
        renderer->initialize(shaderManager.get());
        
        inputSystem = std::make_unique<InputSystem>();
        physicsEngine = std::make_unique<PhysicsEngine>();
        uiSystem = std::make_unique<UISystem>();
        
        // Setup ECS World
        ecsWorld = std::make_unique<ECSWorld>();
        ecsWorld->registerComponent<Transform>();
        ecsWorld->registerComponent<MeshComponent>();
        ecsWorld->registerComponent<CameraComponent>();
        ecsWorld->registerComponent<LightComponent>();
        
        // Create entities
        Entity cubeEntity = ecsWorld->createEntity();
        auto* transform = ecsWorld->addComponent<Transform>(cubeEntity);
        transform->position = Vec3(0, 0, 0);
        transform->rotation = Vec3(0, 0, 0);
        transform->scale = Vec3(1, 1, 1);
        
        Entity lightEntity = ecsWorld->createEntity();
        auto* light = ecsWorld->addComponent<LightComponent>(lightEntity);
        light->color = Vec3(1, 1, 0.9f);
        light->intensity = 5.0f;
        light->isDirectional = false;
        
        // Setup physics for cube
        cubePhysicsIndex = physicsEngine->addBody(Vec3(0, 2, 0), 1.0f, false, 0.5f);
        
        // Add UI elements
        uiSystem->addText(100, 100, "Custom Game Engine v1.0");
        uiSystem->addButton(100, 150, 200, 50, Vec4(0.3f, 0.7f, 1.0f, 1.0f), []() {
            LOGI("Button clicked!");
        });
        
        // Set camera position
        cameraPos = Vec3(0, 2, cameraDistance);
        cameraTarget = Vec3(0, 0, 0);
        
        lastFrameTime = std::chrono::high_resolution_clock::now();
        LOGI("Game Engine initialized successfully");
    }
    
    void initializeEGL() {
        const EGLint attribs[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
            EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
            EGL_BLUE_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_RED_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_DEPTH_SIZE, 24,
            EGL_STENCIL_SIZE, 8,
            EGL_NONE
        };
        
        EGLint attribList[] = {
            EGL_CONTEXT_CLIENT_VERSION, 3,
            EGL_NONE
        };
        
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        eglInitialize(display, nullptr, nullptr);
        
        EGLint numConfigs;
        EGLConfig config;
        eglChooseConfig(display, attribs, &config, 1, &numConfigs);
        
        surface = eglCreateWindowSurface(display, config, app->window, nullptr);
        context = eglCreateContext(display, config, nullptr, attribList);
        
        if(eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
            LOGE("Unable to make EGL current");
            return;
        }
        
        EGLint w, h;
        eglQuerySurface(display, surface, EGL_WIDTH, &w);
        eglQuerySurface(display, surface, EGL_HEIGHT, &h);
        width = w;
        height = h;
        
        glViewport(0, 0, width, height);
        
        LOGI("OpenGL ES 3.0 initialized. Resolution: %dx%d", width, height);
        LOGI("Renderer: %s", glGetString(GL_RENDERER));
        LOGI("Version: %s", glGetString(GL_VERSION));
    }
    
    void update(float deltaTime) {
        // Update physics
        physicsEngine->update(deltaTime);
        
        // Get cube physics position
        Vec3 cubePos = physicsEngine->getPosition(cubePhysicsIndex);
        
        // Update cube transform in ECS
        // In full ECS, we'd use systems to update this
        totalTime += deltaTime;
        
        // Update camera based on input
        Vec2 cameraInput = inputSystem->getCameraRotation();
        cameraPos.x = sinf(cameraInput.x) * cosf(cameraInput.y) * cameraDistance;
        cameraPos.y = sinf(cameraInput.y) * cameraDistance + 2.0f;
        cameraPos.z = cosf(cameraInput.x) * cosf(cameraInput.y) * cameraDistance;
        
        // Clear input events for next frame
        inputSystem->clearEvents();
    }
    
    void render() {
        renderer->clear();
        
        // Setup matrices
        float aspectRatio = (float)width / (float)height;
        Mat4 projection = Mat4::perspective(45.0f, aspectRatio, 0.1f, 100.0f);
        Mat4 view = Mat4::lookAt(cameraPos, cameraTarget, Vec3(0, 1, 0));
        
        // Get cube position from physics
        Vec3 cubePos = physicsEngine->getPosition(cubePhysicsIndex);
        
        // Animate cube
        Mat4 model = Mat4::translation(cubePos);
        model = model * Mat4::rotation(totalTime, Vec3(0, 1, 0));
        model = model * Mat4::rotation(totalTime * 0.5f, Vec3(1, 0, 0));
        model = model * Mat4::scale(Vec3(1, 1, 1));
        
        // Setup light position (orbiting)
        Vec3 lightPos(sinf(totalTime * 0.7f) * 3.0f, 3.0f, cosf(totalTime * 0.7f) * 3.0f);
        
        // Render PBR cube
        renderer->renderCube(model, view, projection, cameraPos, lightPos, totalTime);
        
        // Render UI overlay
        uiSystem->render2DOverlay(width, height);
        
        // Swap buffers
        eglSwapBuffers(display, surface);
    }
    
    void run() {
        while(!app->destroyRequested) {
            // Process Android events
            int events;
            android_poll_source* source;
            while(ALooper_pollAll(0, nullptr, &events, (void**)&source) >= 0) {
                if(source != nullptr) source->process(app, source);
                if(app->destroyRequested) return;
            }
            
            // Calculate deltaTime
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastFrameTime).count();
            lastFrameTime = currentTime;
            
            // Clamp deltaTime
            deltaTime = std::min(deltaTime, 0.1f);
            
            // Game loop
            update(deltaTime);
            render();
        }
    }
    
    void cleanup() {
        if(display != EGL_NO_DISPLAY) {
            eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if(context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
            if(surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
            eglTerminate(display);
        }
        
        renderer->cleanup();
        shaderManager->cleanup();
        
        display = EGL_NO_DISPLAY;
        context = EGL_NO_CONTEXT;
        surface = EGL_NO_SURFACE;
    }
};

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================
void android_main(struct android_app* app) {
    GameEngine engine(app);
    engine.run();
}