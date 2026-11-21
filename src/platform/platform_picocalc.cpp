#include "platform.h"
#include "globals.h"
#include "input.h"
#include "keyboard.h"
#include "status.h"
#include "decoration.h"
#include "benchmark.h"

#include <SDL.h>
#include <alsa/asoundlib.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <png.h>

#include <algorithm>
#include <atomic>
#include <bitset>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct rr_input { SDL_GameController* controller; };
struct rr_input_state {
    rr_button_state_t buttons[RRInputButton_Quit + 1];
    rr_thumb_t sticks[2];
};
struct rr_audio {
    std::atomic<int> volume;
    int frequency;
    int period_frames;
    bool started;
    std::atomic<bool> cancelled;
    std::atomic<bool> paused;
    std::atomic<uint64_t> underruns;
    std::atomic<uint64_t> overruns;
    std::atomic<uint64_t> frames_dropped;
    std::atomic<uint64_t> max_queue_depth;
    std::vector<short> mix_buffer;
    snd_pcm_t* pcm;
};
struct rr_display {
    int fd;
    uint8_t* fb;
    size_t fb_size;
    int width;
    int height;
    int stride;
    int bpp;
    uint32_t red_offset;
    uint32_t green_offset;
    uint32_t blue_offset;
    uint32_t red_length;
    uint32_t green_length;
    uint32_t blue_length;
    int brightness;
};
struct rr_surface {
    rr_display_t* display;
    int width;
    int height;
    int stride;
    uint32_t format;
    uint64_t generation;
    std::vector<uint8_t> pixels;
};
struct rr_presenter {
    rr_display_t* display;
    rr_surface_t* canvas;
    uint32_t background;
    bool loading_wait_completed;
};
struct rr_context { rr_display_t* display; };

static SDL_GameController* active_controller = NULL;
static bool vsync_enabled = false;
static bool vsync_applied = false;
static rr_video_filter_t video_filter = RR_VIDEO_FILTER_DEFAULT;
static rr_video_shader_t video_shader = RR_VIDEO_SHADER_OFF;
static rr_presenter_t* active_presenter = NULL;
static rr_context_t* active_context = NULL;
static rr_display_t* picocalc_display = NULL;
static bool controller_mappings_loaded = false;
static uint64_t next_surface_generation = 1;

static void configure_picocalc_input_hints() {
    static bool configured = false;
    if (configured) return;
    configured = true;
    if (!std::getenv("SDL_VIDEODRIVER"))
        SDL_SetHint("SDL_VIDEODRIVER", "kmsdrm");
    SDL_SetHint("SDL_GAMECONTROLLER_USE_BUTTON_LABELS", "0");
}

static void ensure_sdl(uint32_t flags) {
    if ((SDL_WasInit(flags) & flags) == flags) return;

    if ((flags & SDL_INIT_VIDEO) && !(SDL_WasInit(SDL_INIT_VIDEO) & SDL_INIT_VIDEO)) {
        configure_picocalc_input_hints();
        if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
            std::fprintf(stderr, "RetroRun PicoCalc SDL video init failed: %s\n",
                         SDL_GetError());
            if (!std::getenv("SDL_VIDEODRIVER")) {
                SDL_SetHint("SDL_VIDEODRIVER", "dummy");
                if (SDL_InitSubSystem(SDL_INIT_VIDEO) == 0)
                    std::fprintf(stderr, "RetroRun PicoCalc SDL video using dummy input fallback\n");
                else
                    std::fprintf(stderr, "RetroRun PicoCalc SDL dummy fallback failed: %s\n",
                                 SDL_GetError());
            }
        }
    }

    const uint32_t rest = flags & ~SDL_INIT_VIDEO;
    if (rest && (SDL_WasInit(rest) & rest) != rest) {
        if (SDL_InitSubSystem(rest) != 0)
            std::fprintf(stderr, "RetroRun PicoCalc SDL init 0x%x failed: %s\n",
                         rest, SDL_GetError());
    }
}

static void load_controller_mapping_file(const char* path) {
    if (!path || !*path) return;
    std::FILE* file = std::fopen(path, "rb");
    if (!file) return;
    std::fclose(file);
    const int added = SDL_GameControllerAddMappingsFromFile(path);
    if (added >= 0)
        std::fprintf(stderr, "RetroRun SDL controller DB: loaded %d mapping(s) from %s\n",
                     added, path);
    else
        std::fprintf(stderr, "RetroRun SDL controller DB: could not read %s: %s\n",
                     path, SDL_GetError());
}

static void load_controller_mappings() {
    if (controller_mappings_loaded) return;
    controller_mappings_loaded = true;
    static const char* paths[] = {
        "/storage/.config/SDL-GameControllerDB/gamecontrollerdb.txt",
        "/home/ark/.config/SDL-GameControllerDB/gamecontrollerdb.txt",
        "/usr/share/games/SDL_GameControllerDB/gamecontrollerdb.txt",
        "/usr/share/SDL_GameControllerDB/gamecontrollerdb.txt",
    };
    for (const char* path : paths) load_controller_mapping_file(path);
    load_controller_mapping_file(std::getenv("SDL_GAMECONTROLLERCONFIG_FILE"));
    load_controller_mapping_file(std::getenv("RETRORUN_SDL_CONTROLLER_DB"));
}

static void add_handheld_controller_fallback(int device_index) {
    const char* name = SDL_JoystickNameForIndex(device_index);
    if (!name || std::strcmp(name, "GO-Super Gamepad") != 0) return;

    const SDL_JoystickGUID device_guid = SDL_JoystickGetDeviceGUID(device_index);
    char guid[33] = {};
    SDL_JoystickGetGUIDString(device_guid, guid, static_cast<int>(sizeof(guid)));

    char* existing_mapping = SDL_GameControllerMappingForGUID(device_guid);
    const bool has_start = existing_mapping && std::strstr(existing_mapping, ",start:") != NULL;
    if (has_start) {
        SDL_free(existing_mapping);
        return;
    }
    if (existing_mapping)
        std::fprintf(stderr, "RetroRun SDL controller: replacing incomplete RG351 mapping: %s\n",
                     existing_mapping);
    SDL_free(existing_mapping);

    const std::string mapping =
        std::string(guid) + ",GO-Super Gamepad,"
        "a:b1,b:b0,x:b2,y:b3,back:b12,start:b13,guide:b16,"
        "dpleft:b10,dpdown:b9,dpright:b11,dpup:b8,"
        "leftshoulder:b4,lefttrigger:b6,rightshoulder:b5,righttrigger:b7,"
        "leftstick:b14,rightstick:b15,leftx:a0,lefty:a1,rightx:a2,righty:a3,"
        "platform:Linux,";
    const int result = SDL_GameControllerAddMapping(mapping.c_str());
    std::fprintf(stderr, "RetroRun SDL controller: complete RG351 mapping %s for GUID %s\n",
                 result >= 0 ? "enabled" : "failed", guid);
    if (result < 0)
        std::fprintf(stderr, "RetroRun SDL controller mapping error: %s\n", SDL_GetError());
}

static SDL_GameController* open_controller(int device_index) {
    const char* name = SDL_JoystickNameForIndex(device_index);
    char guid[33] = {};
    SDL_JoystickGetGUIDString(SDL_JoystickGetDeviceGUID(device_index), guid,
                              static_cast<int>(sizeof(guid)));
    add_handheld_controller_fallback(device_index);
    const bool recognized = SDL_IsGameController(device_index) == SDL_TRUE;
    std::fprintf(stderr,
                 "RetroRun SDL joystick[%d]: name='%s', GUID=%s, gamecontroller=%s\n",
                 device_index, name ? name : "unknown", guid,
                 recognized ? "yes" : "no");
    if (!recognized) return NULL;
    SDL_GameController* controller = SDL_GameControllerOpen(device_index);
    if (!controller) {
        std::fprintf(stderr, "RetroRun SDL controller open failed: %s\n", SDL_GetError());
        return NULL;
    }
    char* mapping = SDL_GameControllerMapping(controller);
    const char* controller_name = SDL_GameControllerName(controller);
    std::fprintf(stderr, "RetroRun SDL controller opened: %s%s%s\n",
                 controller_name ? controller_name : "unknown",
                 mapping ? ", mapping=" : "", mapping ? mapping : "");
    SDL_free(mapping);
    return controller;
}

static SDL_GameController* open_first_controller() {
    for (int i = 0; i < SDL_NumJoysticks(); ++i) {
        SDL_GameController* controller = open_controller(i);
        if (controller) return controller;
    }
    return NULL;
}

static void rr_sdl_dispatch_key_event(const SDL_KeyboardEvent* ev);
static void rr_sdl_dispatch_text_event(const SDL_TextInputEvent* ev);

rr_input_t* rr_input_create(const char*) {
    ensure_sdl(SDL_INIT_EVENTS | SDL_INIT_GAMECONTROLLER | SDL_INIT_HAPTIC);
    SDL_StartTextInput();
    load_controller_mappings();
    rr_input_t* input = new rr_input_t();
    input->controller = open_first_controller();
    if (!input->controller)
        std::fprintf(stderr, "RetroRun SDL controller: no compatible gamepad opened\n");
    active_controller = input->controller;
    return input;
}

void rr_input_destroy(rr_input_t* input) {
    if (!input) return;
    if (input->controller) SDL_GameControllerClose(input->controller);
    if (active_controller == input->controller) active_controller = NULL;
    delete input;
}

rr_input_feature_flags_t rr_input_features_get(rr_input_t*) {
    return static_cast<rr_input_feature_flags_t>(RRInputFeatureFlags_Triggers |
                                                  RRInputFeatureFlags_RightAnalog);
}

rr_input_state_t* rr_input_state_create() { return new rr_input_state_t(); }
void rr_input_state_destroy(rr_input_state_t* state) { delete state; }

static void set_key(rr_input_state_t* state, const Uint8* keys, SDL_Scancode key,
                    rr_input_button_t button) {
    if (keys[key]) state->buttons[button] = RRButtonState_Pressed;
}

static float normalized_axis(Sint16 value) {
    return value < 0 ? value / 32768.0f : value / 32767.0f;
}

void rr_input_state_read(rr_input_t* input, rr_input_state_t* state) {
    std::memset(state, 0, sizeof(*state));
    bool controller_change = false;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT)
            state->buttons[RRInputButton_Quit] = RRButtonState_Pressed;
        else if (event.type == SDL_KEYDOWN || event.type == SDL_KEYUP)
            rr_sdl_dispatch_key_event(&event.key);
        else if (event.type == SDL_TEXTINPUT)
            rr_sdl_dispatch_text_event(&event.text);
        else if (event.type == SDL_CONTROLLERDEVICEADDED) {
            std::fprintf(stderr, "RetroRun SDL controller added: device index=%d\n",
                         event.cdevice.which);
            controller_change = true;
        } else if (event.type == SDL_CONTROLLERDEVICEREMOVED && input->controller &&
                   SDL_JoystickInstanceID(SDL_GameControllerGetJoystick(input->controller)) ==
                       event.cdevice.which) {
            std::fprintf(stderr, "RetroRun SDL controller removed: instance=%d\n",
                         event.cdevice.which);
            SDL_GameControllerClose(input->controller);
            input->controller = NULL;
            active_controller = NULL;
            controller_change = true;
        }
    }

    if (input->controller && SDL_GameControllerGetAttached(input->controller) == SDL_FALSE) {
        std::fprintf(stderr, "RetroRun SDL controller detached; reopening input device\n");
        SDL_GameControllerClose(input->controller);
        input->controller = NULL;
        active_controller = NULL;
        controller_change = true;
    }
    if (!input->controller && controller_change) {
        input->controller = open_first_controller();
        active_controller = input->controller;
        if (!input->controller)
            std::fprintf(stderr, "RetroRun SDL controller reconnect failed: no compatible gamepad\n");
    }

    const Uint8* keys = SDL_GetKeyboardState(NULL);
    set_key(state, keys, SDL_SCANCODE_UP, RRInputButton_DPadUp);
    set_key(state, keys, SDL_SCANCODE_DOWN, RRInputButton_DPadDown);
    set_key(state, keys, SDL_SCANCODE_LEFT, RRInputButton_DPadLeft);
    set_key(state, keys, SDL_SCANCODE_RIGHT, RRInputButton_DPadRight);
    set_key(state, keys, SDL_SCANCODE_X, RRInputButton_A);
    set_key(state, keys, SDL_SCANCODE_Z, RRInputButton_B);
    set_key(state, keys, SDL_SCANCODE_S, RRInputButton_X);
    set_key(state, keys, SDL_SCANCODE_A, RRInputButton_Y);
    set_key(state, keys, SDL_SCANCODE_BACKSPACE, RRInputButton_SELECT);
    set_key(state, keys, SDL_SCANCODE_RETURN, RRInputButton_START);
    set_key(state, keys, SDL_SCANCODE_Q, RRInputButton_TopLeft);
    set_key(state, keys, SDL_SCANCODE_W, RRInputButton_TopRight);
    set_key(state, keys, SDL_SCANCODE_1, RRInputButton_TriggerLeft);
    set_key(state, keys, SDL_SCANCODE_2, RRInputButton_TriggerRight);
    set_key(state, keys, SDL_SCANCODE_3, RRInputButton_THUMBL);
    set_key(state, keys, SDL_SCANCODE_4, RRInputButton_THUMBR);
    set_key(state, keys, SDL_SCANCODE_ESCAPE, RRInputButton_Quit);

    SDL_GameController* pad = input->controller;
    if (!pad) return;
    SDL_Joystick* raw_joystick = SDL_GameControllerGetJoystick(pad);
    if (enable_key_log && raw_joystick) {
        static std::vector<Uint8> previous_raw_buttons;
        const int raw_count = SDL_JoystickNumButtons(raw_joystick);
        if (static_cast<int>(previous_raw_buttons.size()) != raw_count)
            previous_raw_buttons.assign(std::max(0, raw_count), 0);
        for (int index = 0; index < raw_count; ++index) {
            const Uint8 pressed = SDL_JoystickGetButton(raw_joystick, index);
            if (pressed && !previous_raw_buttons[index])
                std::fprintf(stderr, "RetroRun SDL raw button: b%d pressed\n", index);
            previous_raw_buttons[index] = pressed;
        }
    }
    const int raw_button_count = raw_joystick ? SDL_JoystickNumButtons(raw_joystick) : 0;
    const char* raw_name = raw_joystick ? SDL_JoystickName(raw_joystick) : NULL;
    const bool rg351_physical_layout = isRG351V() && raw_name &&
        std::strcmp(raw_name, "GO-Super Gamepad") == 0 &&
        raw_button_count == 12 && SDL_JoystickNumHats(raw_joystick) > 0;

    if (rg351_physical_layout && raw_joystick && raw_button_count >= 12) {
        auto raw_button = [&](int index, rr_input_button_t portable) {
            if (SDL_JoystickGetButton(raw_joystick, index))
                state->buttons[portable] = RRButtonState_Pressed;
        };
        raw_button(1, RRInputButton_A);
        raw_button(0, RRInputButton_B);
        raw_button(2, RRInputButton_X);
        raw_button(3, RRInputButton_Y);
        raw_button(7, RRInputButton_F1);
        raw_button(8, RRInputButton_F2);
        raw_button(11, RRInputButton_F3);
        raw_button(10, RRInputButton_F4);
        raw_button(9, RRInputButton_F5);
        raw_button(6, RRInputButton_F6);
        raw_button(4, RRInputButton_TopLeft);
        raw_button(5, RRInputButton_TopRight);
        const Uint8 hat = SDL_JoystickNumHats(raw_joystick) > 0
            ? SDL_JoystickGetHat(raw_joystick, 0) : SDL_HAT_CENTERED;
        if (hat & SDL_HAT_UP) state->buttons[RRInputButton_DPadUp] = RRButtonState_Pressed;
        if (hat & SDL_HAT_DOWN) state->buttons[RRInputButton_DPadDown] = RRButtonState_Pressed;
        if (hat & SDL_HAT_LEFT) state->buttons[RRInputButton_DPadLeft] = RRButtonState_Pressed;
        if (hat & SDL_HAT_RIGHT) state->buttons[RRInputButton_DPadRight] = RRButtonState_Pressed;
        raw_button(8, RRInputButton_THUMBL);
        state->sticks[RRInputThumbstick_Left] = {
            normalized_axis(SDL_JoystickGetAxis(raw_joystick, 0)),
            normalized_axis(SDL_JoystickGetAxis(raw_joystick, 1))};
        state->sticks[RRInputThumbstick_Right] = {
            normalized_axis(SDL_JoystickGetAxis(raw_joystick, 2)),
            normalized_axis(SDL_JoystickGetAxis(raw_joystick, 3))};
        return;
    }

#define PAD_BUTTON(sdl, rr) if (SDL_GameControllerGetButton(pad, sdl)) state->buttons[rr] = RRButtonState_Pressed
    PAD_BUTTON(SDL_CONTROLLER_BUTTON_DPAD_UP, RRInputButton_DPadUp);
    PAD_BUTTON(SDL_CONTROLLER_BUTTON_DPAD_DOWN, RRInputButton_DPadDown);
    PAD_BUTTON(SDL_CONTROLLER_BUTTON_DPAD_LEFT, RRInputButton_DPadLeft);
    PAD_BUTTON(SDL_CONTROLLER_BUTTON_DPAD_RIGHT, RRInputButton_DPadRight);
    const char* mapped_name = SDL_GameControllerName(pad);
    const bool handheld_labels = (mapped_name && std::strcmp(mapped_name, "GO-Super Gamepad") == 0) ||
        (raw_name && std::strcmp(raw_name, "GO-Super Gamepad") == 0);
    if (handheld_labels) {
        PAD_BUTTON(SDL_CONTROLLER_BUTTON_A, RRInputButton_A);
        PAD_BUTTON(SDL_CONTROLLER_BUTTON_B, RRInputButton_B);
        PAD_BUTTON(SDL_CONTROLLER_BUTTON_X, RRInputButton_X);
        PAD_BUTTON(SDL_CONTROLLER_BUTTON_Y, RRInputButton_Y);
    } else {
        PAD_BUTTON(SDL_CONTROLLER_BUTTON_A, RRInputButton_B);
        PAD_BUTTON(SDL_CONTROLLER_BUTTON_B, RRInputButton_A);
        PAD_BUTTON(SDL_CONTROLLER_BUTTON_X, RRInputButton_Y);
        PAD_BUTTON(SDL_CONTROLLER_BUTTON_Y, RRInputButton_X);
    }
    PAD_BUTTON(SDL_CONTROLLER_BUTTON_BACK, RRInputButton_SELECT);
    PAD_BUTTON(SDL_CONTROLLER_BUTTON_START, RRInputButton_START);
    PAD_BUTTON(SDL_CONTROLLER_BUTTON_LEFTSHOULDER, RRInputButton_TopLeft);
    PAD_BUTTON(SDL_CONTROLLER_BUTTON_RIGHTSHOULDER, RRInputButton_TopRight);
    SDL_Joystick* joystick = SDL_GameControllerGetJoystick(pad);
    const int button_count = joystick ? SDL_JoystickNumButtons(joystick) : 0;
    const char* pad_name = SDL_GameControllerName(pad);
    const char* joystick_name = joystick ? SDL_JoystickName(joystick) : NULL;
    const bool go_super =
        (pad_name && std::strcmp(pad_name, "GO-Super Gamepad") == 0) ||
        (joystick_name && std::strcmp(joystick_name, "GO-Super Gamepad") == 0);
    const bool rg351_layout = go_super || isRG351V();
    const bool guide =
        SDL_GameControllerGetButton(pad, SDL_CONTROLLER_BUTTON_GUIDE) != 0 ||
        (rg351_layout && button_count > 16 && SDL_JoystickGetButton(joystick, 16));
    const bool left_stick =
        SDL_GameControllerGetButton(pad, SDL_CONTROLLER_BUTTON_LEFTSTICK) != 0 ||
        (rg351_layout && button_count > 14 && SDL_JoystickGetButton(joystick, 14));
    const bool right_stick =
        SDL_GameControllerGetButton(pad, SDL_CONTROLLER_BUTTON_RIGHTSTICK) != 0 ||
        (rg351_layout && button_count > 15 && SDL_JoystickGetButton(joystick, 15));
    if (isMiniloongPocket1() && joystick && button_count > 10 &&
        SDL_JoystickGetButton(joystick, 10))
        state->buttons[RRInputButton_MENU] = RRButtonState_Pressed;
    const bool left_hotkey = rg351_layout ? (left_stick || guide) : left_stick;
    if (left_hotkey) state->buttons[RRInputButton_THUMBL] = RRButtonState_Pressed;
    if (right_stick) state->buttons[RRInputButton_THUMBR] = RRButtonState_Pressed;
#undef PAD_BUTTON
    const bool left_trigger =
        SDL_GameControllerGetAxis(pad, SDL_CONTROLLER_AXIS_TRIGGERLEFT) > 16000;
    const bool right_trigger =
        SDL_GameControllerGetAxis(pad, SDL_CONTROLLER_AXIS_TRIGGERRIGHT) > 16000;
    if (left_trigger) state->buttons[RRInputButton_TriggerLeft] = RRButtonState_Pressed;
    if (right_trigger) state->buttons[RRInputButton_TriggerRight] = RRButtonState_Pressed;
    if (isRG351V()) {
        state->buttons[RRInputButton_F1] = state->buttons[RRInputButton_SELECT];
        state->buttons[RRInputButton_F6] = state->buttons[RRInputButton_START];
        state->buttons[RRInputButton_F4] = state->buttons[RRInputButton_TriggerLeft];
        state->buttons[RRInputButton_F3] = state->buttons[RRInputButton_TriggerRight];
        if (left_hotkey) state->buttons[RRInputButton_F2] = RRButtonState_Pressed;
        if (right_stick) state->buttons[RRInputButton_F5] = RRButtonState_Pressed;
    }
    state->sticks[RRInputThumbstick_Left] = {
        normalized_axis(SDL_GameControllerGetAxis(pad, SDL_CONTROLLER_AXIS_LEFTX)),
        normalized_axis(SDL_GameControllerGetAxis(pad, SDL_CONTROLLER_AXIS_LEFTY))};
    state->sticks[RRInputThumbstick_Right] = {
        normalized_axis(SDL_GameControllerGetAxis(pad, SDL_CONTROLLER_AXIS_RIGHTX)),
        normalized_axis(SDL_GameControllerGetAxis(pad, SDL_CONTROLLER_AXIS_RIGHTY))};
}

rr_button_state_t rr_input_state_button_get(rr_input_state_t* state, rr_input_button_t button) {
    return state && button >= 0 && button <= RRInputButton_Quit
        ? state->buttons[button] : RRButtonState_Released;
}
void rr_input_state_button_set(rr_input_state_t* state, rr_input_button_t button,
                               rr_button_state_t value) {
    if (state && button >= 0 && button <= RRInputButton_Quit)
        state->buttons[button] = value;
}
rr_thumb_t rr_input_state_thumbstick_get(rr_input_state_t* state, rr_input_thumbstick_t stick) {
    return state ? state->sticks[stick] : rr_thumb_t{0.0f, 0.0f};
}
void rr_input_state_thumbstick_set_null(rr_input_state_t* state, rr_input_thumbstick_t stick) {
    if (state) state->sticks[stick] = {0.0f, 0.0f};
}
void rr_input_battery_read(rr_input_t*, rr_battery_state_t* state) {
    state->level = 100;
    state->status = RRBattery_Status_Unknown;
}
void rr_input_brightness_read(rr_input_t*, rr_brightness_state_t* state) { state->level = 100; }
void rr_input_brightness_write(int) {}
bool rr_input_set_rumble(uint16_t low, uint16_t high, uint32_t duration) {
    return active_controller &&
           SDL_GameControllerRumble(active_controller, low, high, duration) == 0;
}

rr_audio_t* rr_audio_create(int frequency) {
    snd_pcm_t* pcm = nullptr;
    int err = snd_pcm_open(&pcm, "default", SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) {
        std::fprintf(stderr, "RetroRun ALSA audio open failed: %s\n", snd_strerror(err));
        return nullptr;
    }
    err = snd_pcm_set_params(pcm, SND_PCM_FORMAT_S16_LE,
                             SND_PCM_ACCESS_RW_INTERLEAVED, 2,
                             static_cast<unsigned>(frequency), 1, 100000);
    if (err < 0) {
        std::fprintf(stderr, "RetroRun ALSA audio setup failed: %s\n", snd_strerror(err));
        snd_pcm_close(pcm);
        return nullptr;
    }
    snd_pcm_nonblock(pcm, 1);

    snd_pcm_uframes_t buffer_frames = 0;
    snd_pcm_uframes_t period_frames = 0;
    snd_pcm_get_params(pcm, &buffer_frames, &period_frames);

    rr_audio_t* audio = new rr_audio_t();
    audio->pcm = pcm;
    audio->volume = 100;
    audio->frequency = frequency;
    audio->period_frames = static_cast<int>(period_frames);
    audio->started = false;
    audio->cancelled = false;
    audio->paused = false;
    audio->underruns = 0;
    audio->overruns = 0;
    audio->frames_dropped = 0;
    audio->max_queue_depth = 0;
    std::fprintf(stderr,
                 "RetroRun ALSA audio: device=default, frequency=%d, period=%lu, buffer=%lu\n",
                 frequency, static_cast<unsigned long>(period_frames),
                 static_cast<unsigned long>(buffer_frames));
    return audio;
}

void rr_audio_destroy(rr_audio_t* audio) {
    if (!audio) return;
    if (audio->pcm) {
        snd_pcm_drop(audio->pcm);
        snd_pcm_close(audio->pcm);
    }
    delete audio;
}

void rr_audio_release_thread(rr_audio_t*) {}

bool rr_audio_submit(rr_audio_t* audio, const short* data, int frames) {
    if (!audio || !audio->pcm || !data || frames <= 0 ||
        audio->cancelled.load(std::memory_order_relaxed) ||
        audio->paused.load(std::memory_order_relaxed))
        return false;

    const short* source = data;
    const int volume = audio->volume.load(std::memory_order_relaxed);
    if (volume < 100) {
        audio->mix_buffer.resize(static_cast<size_t>(frames) * 2);
        for (size_t i = 0; i < audio->mix_buffer.size(); ++i)
            audio->mix_buffer[i] = static_cast<short>((data[i] * volume) / 100);
        source = audio->mix_buffer.data();
    }

    int remaining = frames;
    const short* cursor = source;
    while (remaining > 0) {
        const snd_pcm_sframes_t written =
            snd_pcm_writei(audio->pcm, cursor, static_cast<snd_pcm_uframes_t>(remaining));
        if (written == -EAGAIN) {
            audio->overruns.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (written < 0) {
            if (written == -EPIPE)
                audio->underruns.fetch_add(1, std::memory_order_relaxed);
            const int recovered = snd_pcm_recover(audio->pcm, static_cast<int>(written), 1);
            if (recovered < 0) {
                std::fprintf(stderr, "RetroRun ALSA audio write failed: %s\n",
                             snd_strerror(recovered));
                audio->frames_dropped.fetch_add(static_cast<uint64_t>(frames),
                                                std::memory_order_relaxed);
                return false;
            }
            continue;
        }
        if (written == 0) {
            audio->overruns.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        cursor += written * 2;
        remaining -= static_cast<int>(written);
    }
    if (!audio->started) {
        const int start_error = snd_pcm_start(audio->pcm);
        if (start_error < 0 && start_error != -EBADFD)
            std::fprintf(stderr, "RetroRun ALSA audio start failed: %s\n",
                         snd_strerror(start_error));
        audio->started = true;
    }
    return true;
}

bool rr_audio_valid(rr_audio_t* audio) { return audio && audio->pcm; }

void rr_audio_flush(rr_audio_t* audio) {
    if (!audio || !audio->pcm) return;
    snd_pcm_drop(audio->pcm);
    snd_pcm_prepare(audio->pcm);
    audio->started = false;
}

void rr_audio_pause(rr_audio_t* audio, bool paused) {
    if (!audio || !audio->pcm) return;
    audio->paused.store(paused, std::memory_order_relaxed);
    audio->started = false;
    if (paused)
        snd_pcm_drop(audio->pcm);
    else
        snd_pcm_prepare(audio->pcm);
}

void rr_audio_cancel(rr_audio_t* audio) {
    if (!audio) return;
    audio->cancelled.store(true, std::memory_order_relaxed);
    if (audio->pcm) snd_pcm_drop(audio->pcm);
}

void rr_audio_diagnostics_get(rr_audio_t* audio, rr_audio_diagnostics_t* diagnostics) {
    if (!diagnostics) return;
    *diagnostics = {};
    if (!audio) return;
    diagnostics->buffer_underruns = audio->underruns.load();
    diagnostics->buffer_overruns = audio->overruns.load();
    diagnostics->frames_dropped = audio->frames_dropped.load();
    diagnostics->max_queue_depth = audio->max_queue_depth.load();
}

uint32_t rr_audio_volume_get(rr_audio_t* audio, const char*) {
    return audio ? audio->volume.load(std::memory_order_relaxed) : 0;
}
void rr_audio_volume_set(rr_audio_t* audio, uint32_t value, const char*) {
    if (audio)
        audio->volume.store(std::min<uint32_t>(value, 100), std::memory_order_relaxed);
}

void rr_platform_preinit() {
    ensure_sdl(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
    if (!rr_display_create())
        std::fprintf(stderr, "RetroRun PicoCalc framebuffer preallocation failed\n");
    std::fprintf(stderr, "RetroRun PicoCalc preinitialized: sdl_video=%s\n",
                 SDL_GetCurrentVideoDriver() ? SDL_GetCurrentVideoDriver() : "unknown");
}

int rr_pixel_format_bpp(uint32_t format) {
    switch (format) {
    case RR_PIXEL_FORMAT_RGB565:
    case RR_PIXEL_FORMAT_RGBA5551:
        return 16;
    case RR_PIXEL_FORMAT_RGB888:
        return 24;
    default:
        return 32;
    }
}

static bool known_pixel_format(uint32_t format) {
    return format == RR_PIXEL_FORMAT_RGB565 || format == RR_PIXEL_FORMAT_RGB888 ||
           format == RR_PIXEL_FORMAT_XRGB8888 || format == RR_PIXEL_FORMAT_RGBA8888 ||
           format == RR_PIXEL_FORMAT_RGBA5551;
}

static inline uint16_t make_rgb565(uint8_t r, uint8_t g, uint8_t b) {
    return static_cast<uint16_t>(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
}

static inline void decode_pixel(const rr_surface_t* surface, int x, int y,
                                uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) {
    const uint8_t* row = surface->pixels.data() + static_cast<size_t>(y) * surface->stride;
    if (surface->format == RR_PIXEL_FORMAT_RGB565) {
        uint16_t value = 0;
        std::memcpy(&value, row + x * 2, sizeof(value));
        r = static_cast<uint8_t>(((value >> 11) & 0x1f) * 255 / 31);
        g = static_cast<uint8_t>(((value >> 5) & 0x3f) * 255 / 63);
        b = static_cast<uint8_t>((value & 0x1f) * 255 / 31);
        a = 255;
    } else if (surface->format == RR_PIXEL_FORMAT_RGBA5551) {
        uint16_t value = 0;
        std::memcpy(&value, row + x * 2, sizeof(value));
        r = static_cast<uint8_t>(((value >> 10) & 0x1f) * 255 / 31);
        g = static_cast<uint8_t>(((value >> 5) & 0x1f) * 255 / 31);
        b = static_cast<uint8_t>((value & 0x1f) * 255 / 31);
        a = (value & 0x8000) ? 255 : 0;
    } else if (surface->format == RR_PIXEL_FORMAT_RGB888) {
        r = row[x * 3 + 0];
        g = row[x * 3 + 1];
        b = row[x * 3 + 2];
        a = 255;
    } else if (surface->format == RR_PIXEL_FORMAT_RGBA8888) {
        r = row[x * 4 + 0];
        g = row[x * 4 + 1];
        b = row[x * 4 + 2];
        a = row[x * 4 + 3];
    } else {
        r = row[x * 4 + 2];
        g = row[x * 4 + 1];
        b = row[x * 4 + 0];
        a = 255;
    }
}

static inline void encode_pixel(rr_surface_t* surface, int x, int y,
                                uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    uint8_t* row = surface->pixels.data() + static_cast<size_t>(y) * surface->stride;
    if (surface->format == RR_PIXEL_FORMAT_RGB565) {
        const uint16_t value = make_rgb565(r, g, b);
        std::memcpy(row + x * 2, &value, sizeof(value));
    } else if (surface->format == RR_PIXEL_FORMAT_RGBA5551) {
        const uint16_t value = static_cast<uint16_t>(
            (a ? 0x8000 : 0) | ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3));
        std::memcpy(row + x * 2, &value, sizeof(value));
    } else if (surface->format == RR_PIXEL_FORMAT_RGB888) {
        row[x * 3 + 0] = r;
        row[x * 3 + 1] = g;
        row[x * 3 + 2] = b;
    } else if (surface->format == RR_PIXEL_FORMAT_RGBA8888) {
        row[x * 4 + 0] = r;
        row[x * 4 + 1] = g;
        row[x * 4 + 2] = b;
        row[x * 4 + 3] = a;
    } else {
        row[x * 4 + 0] = b;
        row[x * 4 + 1] = g;
        row[x * 4 + 2] = r;
        row[x * 4 + 3] = 0;
    }
}

static void blit_internal(rr_surface_t* source, int sx, int sy, int sw, int sh,
                          rr_surface_t* dest, int dx, int dy, int dw, int dh,
                          rr_rotation_t rotation, uint8_t opacity,
                          const int* clip) {
    if (!source || !dest || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;
    int x0 = dx;
    int y0 = dy;
    int x1 = dx + dw;
    int y1 = dy + dh;
    if (clip) {
        x0 = std::max(x0, clip[0]);
        y0 = std::max(y0, clip[1]);
        x1 = std::min(x1, clip[0] + clip[2]);
        y1 = std::min(y1, clip[1] + clip[3]);
    }
    x0 = std::max(x0, 0);
    y0 = std::max(y0, 0);
    x1 = std::min(x1, dest->width);
    y1 = std::min(y1, dest->height);
    if (x0 >= x1 || y0 >= y1) return;

    if (rotation == RR_ROTATION_DEGREES_0 && opacity == 255 &&
        source->format == RR_PIXEL_FORMAT_RGB565 &&
        dest->format == RR_PIXEL_FORMAT_RGB565) {
        const uint32_t x_step = (static_cast<uint32_t>(sw) << 16) / static_cast<uint32_t>(dw);
        const uint32_t y_step = (static_cast<uint32_t>(sh) << 16) / static_cast<uint32_t>(dh);
        uint32_t source_y = (static_cast<uint32_t>(sy) << 16) +
                            static_cast<uint32_t>(y0 - dy) * y_step;
        const uint16_t* source_pixels = reinterpret_cast<const uint16_t*>(source->pixels.data());
        uint16_t* dest_pixels = reinterpret_cast<uint16_t*>(dest->pixels.data());
        const int source_pitch = source->stride / 2;
        const int dest_pitch = dest->stride / 2;
        for (int y = y0; y < y1; ++y) {
            const uint16_t* source_row = source_pixels + (source_y >> 16) * source_pitch;
            uint16_t* dest_row = dest_pixels + y * dest_pitch;
            uint32_t source_x = (static_cast<uint32_t>(sx) << 16) +
                                static_cast<uint32_t>(x0 - dx) * x_step;
            for (int x = x0; x < x1; ++x) {
                dest_row[x] = source_row[source_x >> 16];
                source_x += x_step;
            }
            source_y += y_step;
        }
        dest->generation = next_surface_generation++;
        return;
    }

    const bool source_alpha = source->format == RR_PIXEL_FORMAT_RGBA8888 ||
                              source->format == RR_PIXEL_FORMAT_RGBA5551;
    for (int y = y0; y < y1; ++y) {
        const int oy = y - dy;
        for (int x = x0; x < x1; ++x) {
            const int ox = x - dx;
            int ix = 0;
            int iy = 0;
            switch (rotation) {
            case RR_ROTATION_DEGREES_90:
                ix = sx + oy * sw / dh;
                iy = sy + sh - 1 - ox * sh / dw;
                break;
            case RR_ROTATION_DEGREES_180:
                ix = sx + sw - 1 - ox * sw / dw;
                iy = sy + sh - 1 - oy * sh / dh;
                break;
            case RR_ROTATION_DEGREES_270:
                ix = sx + sw - 1 - oy * sw / dh;
                iy = sy + ox * sh / dw;
                break;
            case RR_ROTATION_HORIZONTAL:
                ix = sx + sw - 1 - ox * sw / dw;
                iy = sy + oy * sh / dh;
                break;
            case RR_ROTATION_VERTICAL:
                ix = sx + ox * sw / dw;
                iy = sy + sh - 1 - oy * sh / dh;
                break;
            default:
                ix = sx + ox * sw / dw;
                iy = sy + oy * sh / dh;
                break;
            }
            ix = std::clamp(ix, sx, sx + sw - 1);
            iy = std::clamp(iy, sy, sy + sh - 1);

            uint8_t r = 0, g = 0, b = 0, a = 255;
            decode_pixel(source, ix, iy, r, g, b, a);
            if (source_alpha)
                a = static_cast<uint8_t>((static_cast<unsigned>(a) * opacity) / 255);
            else
                a = opacity;
            if (a == 0) continue;
            if (a == 255) {
                encode_pixel(dest, x, y, r, g, b, 255);
            } else {
                uint8_t dr = 0, dg = 0, db = 0, da = 255;
                decode_pixel(dest, x, y, dr, dg, db, da);
                const unsigned inv = 255 - a;
                encode_pixel(dest, x, y,
                             static_cast<uint8_t>((r * a + dr * inv) / 255),
                             static_cast<uint8_t>((g * a + dg * inv) / 255),
                             static_cast<uint8_t>((b * a + db * inv) / 255), 255);
            }
        }
    }
    dest->generation = next_surface_generation++;
}

static void fb_write_canvas(rr_display_t* display, rr_surface_t* canvas) {
    if (!display || !display->fb || !canvas) return;
    const int copy_width = std::min(display->width, canvas->width);
    const int copy_height = std::min(display->height, canvas->height);
    if (display->bpp == 16 && display->red_offset == 11 &&
        display->green_offset == 5 && display->blue_offset == 0 &&
        display->red_length == 5 && display->green_length == 6 &&
        display->blue_length == 5) {
        for (int y = 0; y < copy_height; ++y) {
            std::memcpy(display->fb + static_cast<size_t>(y) * display->stride,
                        canvas->pixels.data() + static_cast<size_t>(y) * canvas->stride,
                        static_cast<size_t>(copy_width) * 2);
        }
        return;
    }

    for (int y = 0; y < copy_height; ++y) {
        uint8_t* out = display->fb + static_cast<size_t>(y) * display->stride;
        for (int x = 0; x < copy_width; ++x) {
            uint8_t r = 0, g = 0, b = 0, a = 255;
            decode_pixel(canvas, x, y, r, g, b, a);
            if (display->bpp == 16) {
                const unsigned rr = display->red_length
                    ? (static_cast<unsigned>(r) * ((1u << display->red_length) - 1) / 255) : 0;
                const unsigned gg = display->green_length
                    ? (static_cast<unsigned>(g) * ((1u << display->green_length) - 1) / 255) : 0;
                const unsigned bb = display->blue_length
                    ? (static_cast<unsigned>(b) * ((1u << display->blue_length) - 1) / 255) : 0;
                const uint16_t value = static_cast<uint16_t>((rr << display->red_offset) |
                                                             (gg << display->green_offset) |
                                                             (bb << display->blue_offset));
                std::memcpy(out + x * 2, &value, sizeof(value));
            } else if (display->bpp == 32) {
                const unsigned rr = display->red_length
                    ? (static_cast<unsigned>(r) * ((1u << display->red_length) - 1) / 255) : 0;
                const unsigned gg = display->green_length
                    ? (static_cast<unsigned>(g) * ((1u << display->green_length) - 1) / 255) : 0;
                const unsigned bb = display->blue_length
                    ? (static_cast<unsigned>(b) * ((1u << display->blue_length) - 1) / 255) : 0;
                const uint32_t value = (rr << display->red_offset) |
                                       (gg << display->green_offset) |
                                       (bb << display->blue_offset);
                std::memcpy(out + x * 4, &value, sizeof(value));
            }
        }
    }
}

rr_display_t* rr_display_create() {
    if (picocalc_display) return picocalc_display;
    ensure_sdl(SDL_INIT_VIDEO);

    const char* path = std::getenv("RETRORUN_FB");
    if (!path || !*path) path = "/dev/fb0";
    const int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "RetroRun PicoCalc framebuffer open failed: %s\n", path);
        return NULL;
    }

    fb_var_screeninfo var = {};
    fb_fix_screeninfo fix = {};
    if (ioctl(fd, FBIOGET_VSCREENINFO, &var) < 0 ||
        ioctl(fd, FBIOGET_FSCREENINFO, &fix) < 0) {
        std::fprintf(stderr, "RetroRun PicoCalc framebuffer query failed\n");
        close(fd);
        return NULL;
    }
    if (var.bits_per_pixel != 16 && var.bits_per_pixel != 32) {
        std::fprintf(stderr, "RetroRun PicoCalc unsupported framebuffer depth %u\n",
                     var.bits_per_pixel);
        close(fd);
        return NULL;
    }

    const int stride = fix.line_length ? static_cast<int>(fix.line_length)
                                       : static_cast<int>(var.xres * var.bits_per_pixel / 8);
    const size_t size = static_cast<size_t>(stride) * var.yres;
    void* mapped = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        std::fprintf(stderr, "RetroRun PicoCalc framebuffer mmap failed\n");
        close(fd);
        return NULL;
    }

    rr_display_t* display = new rr_display_t();
    display->fd = fd;
    display->fb = static_cast<uint8_t*>(mapped);
    display->fb_size = size;
    display->width = static_cast<int>(var.xres);
    display->height = static_cast<int>(var.yres);
    display->stride = stride;
    display->bpp = static_cast<int>(var.bits_per_pixel);
    display->red_offset = var.red.offset;
    display->green_offset = var.green.offset;
    display->blue_offset = var.blue.offset;
    display->red_length = var.red.length;
    display->green_length = var.green.length;
    display->blue_length = var.blue.length;
    display->brightness = 100;
    std::memset(display->fb, 0, display->fb_size);
    picocalc_display = display;
    std::fprintf(stderr,
                 "RetroRun PicoCalc framebuffer: %s %dx%d %dbpp stride=%d rgb=%u/%u/%u\n",
                 path, display->width, display->height, display->bpp,
                 display->stride, display->red_offset, display->green_offset,
                 display->blue_offset);
    return display;
}

void rr_display_destroy(rr_display_t* display) {
    if (!display) return;
    if (picocalc_display == display) picocalc_display = NULL;
    if (display->fb) {
        std::memset(display->fb, 0, display->fb_size);
        munmap(display->fb, display->fb_size);
    }
    if (display->fd >= 0) close(display->fd);
    delete display;
}

int rr_display_width_get(rr_display_t* display) { return display ? display->width : 0; }
int rr_display_height_get(rr_display_t* display) { return display ? display->height : 0; }

void rr_display_diagnostics_get(rr_display_t* display,
                                rr_display_diagnostics_t* diagnostics) {
    if (!diagnostics) return;
    *diagnostics = {};
    if (!display) return;
    diagnostics->available = true;
    std::snprintf(diagnostics->driver, sizeof(diagnostics->driver), "fbdev");
    diagnostics->mode_width = display->width;
    diagnostics->mode_height = display->height;
    diagnostics->page_flip_fallback = true;
}

void rr_display_diagnostics_reset(rr_display_t*) {}

uint32_t rr_display_backlight_get(rr_display_t* display) {
    return display ? static_cast<uint32_t>(display->brightness) : 0;
}
void rr_display_backlight_set(rr_display_t* display, uint32_t value) {
    if (display) display->brightness = static_cast<int>(std::min<uint32_t>(value, 100));
}

rr_surface_t* rr_surface_create(rr_display_t* display, int width, int height,
                                uint32_t format) {
    if (width <= 0 || height <= 0 || !known_pixel_format(format)) return NULL;
    rr_surface_t* surface = new rr_surface_t();
    surface->display = display;
    surface->width = width;
    surface->height = height;
    surface->format = format;
    surface->generation = next_surface_generation++;
    surface->stride = width * (rr_pixel_format_bpp(format) / 8);
    surface->pixels.resize(static_cast<size_t>(surface->stride) * height);
    return surface;
}
void rr_surface_destroy(rr_surface_t* surface) { delete surface; }
int rr_surface_width_get(rr_surface_t* surface) { return surface ? surface->width : 0; }
int rr_surface_height_get(rr_surface_t* surface) { return surface ? surface->height : 0; }
uint32_t rr_surface_format_get(rr_surface_t* surface) {
    return surface ? surface->format : RR_PIXEL_FORMAT_RGB565;
}
int rr_surface_stride_get(rr_surface_t* surface) { return surface ? surface->stride : 0; }
void* rr_surface_map(rr_surface_t* surface) {
    if (!surface) return NULL;
    surface->generation = next_surface_generation++;
    return surface->pixels.data();
}
void rr_surface_unmap(rr_surface_t*) {}

void rr_surface_blit(rr_surface_t* source, int sx, int sy, int sw, int sh,
                     rr_surface_t* dest, int dx, int dy, int dw, int dh,
                     rr_rotation_t rotation) {
    blit_internal(source, sx, sy, sw, sh, dest, dx, dy, dw, dh, rotation, 255, NULL);
}
void rr_surface_blit_pixels(rr_surface_t* source, int sx, int sy, int sw, int sh,
                            rr_surface_t* dest, int dx, int dy, int dw, int dh) {
    blit_internal(source, sx, sy, sw, sh, dest, dx, dy, dw, dh,
                  RR_ROTATION_DEGREES_0, 255, NULL);
}

int rr_surface_save_as_png(rr_surface_t* surface, const char* filename) {
    if (!surface || !filename || !*filename) return -1;
    std::FILE* file = std::fopen(filename, "wb");
    if (!file) return -1;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info = png ? png_create_info_struct(png) : NULL;
    if (!png || !info || setjmp(png_jmpbuf(png))) {
        if (png) png_destroy_write_struct(&png, info ? &info : NULL);
        std::fclose(file);
        return -1;
    }
    png_init_io(png, file);
    png_set_IHDR(png, info, surface->width, surface->height, 8,
                 PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    std::vector<uint8_t> row(static_cast<size_t>(surface->width) * 4);
    for (int y = 0; y < surface->height; ++y) {
        for (int x = 0; x < surface->width; ++x) {
            uint8_t r = 0, g = 0, b = 0, a = 255;
            decode_pixel(surface, x, y, r, g, b, a);
            row[static_cast<size_t>(x) * 4 + 0] = r;
            row[static_cast<size_t>(x) * 4 + 1] = g;
            row[static_cast<size_t>(x) * 4 + 2] = b;
            row[static_cast<size_t>(x) * 4 + 3] = a;
        }
        png_write_row(png, row.data());
    }
    png_write_end(png, info);
    png_destroy_write_struct(&png, &info);
    std::fclose(file);
    return 0;
}

static uint16_t presenter_background(rr_presenter_t* presenter) {
    const uint32_t value = presenter ? presenter->background : 0;
    return make_rgb565(static_cast<uint8_t>((value >> 16) & 0xff),
                       static_cast<uint8_t>((value >> 8) & 0xff),
                       static_cast<uint8_t>(value & 0xff));
}

static void presenter_clear(rr_presenter_t* presenter) {
    if (!presenter || !presenter->canvas) return;
    const uint16_t color = presenter_background(presenter);
    auto* pixels = reinterpret_cast<uint16_t*>(presenter->canvas->pixels.data());
    const size_t count = presenter->canvas->pixels.size() / sizeof(uint16_t);
    std::fill(pixels, pixels + count, color);
    presenter->canvas->generation = next_surface_generation++;
}

static void presenter_present(rr_presenter_t* presenter) {
    if (!presenter || !presenter->display || !presenter->canvas) return;
    fb_write_canvas(presenter->display, presenter->canvas);
}

rr_presenter_t* rr_presenter_create(rr_display_t* display, uint32_t, uint32_t background) {
    if (!display) return NULL;
    rr_presenter_t* presenter = new rr_presenter_t();
    presenter->display = display;
    presenter->canvas = rr_surface_create(display, display->width, display->height,
                                          RR_PIXEL_FORMAT_RGB565);
    presenter->background = background;
    presenter->loading_wait_completed = false;
    if (!presenter->canvas) {
        delete presenter;
        return NULL;
    }
    presenter_clear(presenter);
    active_presenter = presenter;
    return presenter;
}

void rr_presenter_destroy(rr_presenter_t* presenter) {
    if (!presenter) return;
    if (active_presenter == presenter) active_presenter = NULL;
    if (presenter->canvas) rr_surface_destroy(presenter->canvas);
    delete presenter;
}

void rr_presenter_drain(rr_presenter_t*) {}

void rr_presenter_post(rr_presenter_t* presenter, rr_surface_t* surface,
                       int sx, int sy, int sw, int sh,
                       int dx, int dy, int dw, int dh, rr_rotation_t rotation) {
    if (!presenter || !presenter->canvas || !surface) return;
    const uint64_t generation = benchmark_capture_generation();
    const auto started = generation ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
    presenter_clear(presenter);
    blit_internal(surface, sx, sy, sw, sh, presenter->canvas, dx, dy, dw, dh,
                  rotation, 255, NULL);
    presenter_present(presenter);
    if (generation)
        benchmark_presentation_completed(BenchmarkPresentation::Fallback,
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - started).count(), generation);
}

bool rr_presenter_post_direct(rr_presenter_t*, rr_surface_t*, int, int, int, int,
                              int, int, int, int, rr_rotation_t) { return false; }
void rr_presenter_direct_disable(rr_presenter_t*) {}

void rr_presenter_black(rr_presenter_t* presenter, int, int, int, int, rr_rotation_t) {
    if (!presenter || !presenter->canvas) return;
    presenter_clear(presenter);
    presenter_present(presenter);
}

void rr_presenter_wait_for_loading_screen(rr_presenter_t* presenter, unsigned milliseconds) {
    if (!presenter || presenter->loading_wait_completed) return;
    presenter->loading_wait_completed = true;
    const Uint32 started = SDL_GetTicks();
    while (static_cast<Uint32>(SDL_GetTicks() - started) < milliseconds) {
        SDL_PumpEvents();
        SDL_Delay(5);
    }
}

void rr_presenter_post_multiple(rr_presenter_t* presenter, rr_surface_t* base, status* o,
                                int sx, int sy, int sw, int sh,
                                int dx, int dy, int dw, int dh,
                                rr_rotation_t rotation, rr_rotation_t overlay_rotation,
                                bool) {
    if (!presenter || !presenter->canvas || !base || !o) return;
    const uint64_t generation = benchmark_capture_generation();
    const auto started = generation ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
    presenter_clear(presenter);
    const int width = presenter->display->width;
    const int height = presenter->display->height;

    if (o->show_decoration && o->decoration) {
        rr_surface_t* background = decoration_background_surface();
        if (background)
            blit_internal(background, 0, 0, background->width, background->height,
                          presenter->canvas, 0, 0, width, height,
                          RR_ROTATION_DEGREES_0, 255, NULL);
    }
    blit_internal(base, sx, sy, sw, sh, presenter->canvas, dx, dy, dw, dh,
                  rotation, 255, NULL);
    if (o->show_decoration && o->decoration &&
        o->decoration->format == RR_PIXEL_FORMAT_RGBA8888) {
        const int clip[4] = {dx, dy, dw, dh};
        blit_internal(o->decoration, 0, 0, o->decoration->width, o->decoration->height,
                      presenter->canvas, 0, 0, width, height,
                      RR_ROTATION_DEGREES_0, 255, clip);
    }

    const int scale = std::max(1, width / 640);
    auto overlay = [&](rr_surface_t* surface, int x, int y, int w, int h) {
        if (surface)
            blit_internal(surface, 0, 0, surface->width, surface->height,
                          presenter->canvas, x, y, w, h,
                          RR_ROTATION_DEGREES_0, 255, NULL);
    };
    if (o->show_full && o->full)
        blit_internal(o->full, 0, 0, o->full->width, o->full->height,
                      presenter->canvas, dx, dy, dw, dh,
                      overlay_rotation, o->full_opacity, NULL);
    if (o->show_top_left && o->top_left)
        overlay(o->top_left, 0, 0, o->top_left->width * scale, o->top_left->height * scale);
    if (o->show_top_right && o->top_right)
        overlay(o->top_right, width - o->top_right->width * scale, 0,
                o->top_right->width * scale, o->top_right->height * scale);
    if (o->show_bottom_left && o->bottom_left)
        overlay(o->bottom_left, 0, height - o->bottom_left->height * scale,
                o->bottom_left->width * scale, o->bottom_left->height * scale);
    if (o->show_bottom_right && o->bottom_right)
        overlay(o->bottom_right, width - o->bottom_right->width * scale,
                height - o->bottom_right->height * scale,
                o->bottom_right->width * scale, o->bottom_right->height * scale);
    if (o->show_bottom_center && o->bottom_center)
        overlay(o->bottom_center, (width - o->bottom_center->width * scale) / 2,
                height - o->bottom_center->height * scale,
                o->bottom_center->width * scale, o->bottom_center->height * scale);

    presenter_present(presenter);
    if (generation)
        benchmark_presentation_completed(BenchmarkPresentation::Fallback,
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - started).count(), generation);
}

rr_context_t* rr_context_create(rr_display_t*, int, int, const rr_context_attributes_t*) {
    std::fprintf(stderr,
                 "RetroRun PicoCalc framebuffer backend is software-only; "
                 "hardware rendering contexts are unavailable\n");
    return NULL;
}
void rr_context_destroy(rr_context_t* context) {
    if (active_context == context) active_context = NULL;
    delete context;
}
void rr_context_make_current(rr_context_t*) {}
void rr_context_swap_buffers(rr_context_t*, int, int, int, int, int, int,
                             status*, rr_rotation_t, bool) {}
uintptr_t rr_context_framebuffer_get(rr_context_t*) { return 0; }
rr_surface_t* rr_context_surface_last(rr_context_t*) { return nullptr; }
rr_surface_t* rr_context_surface_lock(rr_context_t*) { return NULL; }
void rr_context_surface_unlock(rr_context_t*, rr_surface_t* surface) { delete surface; }
void* rr_context_get_proc_address(const char*) { return NULL; }

void rr_video_sync() {}
bool rr_video_vsync_set(bool) {
    vsync_enabled = false;
    vsync_applied = false;
    return false;
}
bool rr_video_vsync_get() { return vsync_enabled; }
bool rr_video_vsync_applied() { return vsync_applied; }
void rr_video_filter_set(rr_video_filter_t filter) { video_filter = filter; }
rr_video_filter_t rr_video_filter_get() { return video_filter; }
void rr_video_shader_set(rr_video_shader_t shader) { video_shader = shader; }
rr_video_shader_t rr_video_shader_get() { return video_shader; }

const char* rr_platform_backend_name() { return "picocalc-fb"; }
const char* rr_platform_renderer_name() { return "fbdev-rgb565"; }

uint32_t rr_platform_capabilities() {
    return RRPlatformCapability_PhysicalKeyboard;
}

static bool is_frontend_hotkey(SDL_Scancode sc) {
    switch (sc) {
    case SDL_SCANCODE_UP: case SDL_SCANCODE_DOWN:
    case SDL_SCANCODE_LEFT: case SDL_SCANCODE_RIGHT:
    case SDL_SCANCODE_X: case SDL_SCANCODE_Z:
    case SDL_SCANCODE_S: case SDL_SCANCODE_A:
    case SDL_SCANCODE_BACKSPACE: case SDL_SCANCODE_RETURN:
    case SDL_SCANCODE_Q: case SDL_SCANCODE_W:
    case SDL_SCANCODE_1: case SDL_SCANCODE_2:
    case SDL_SCANCODE_3: case SDL_SCANCODE_4:
    case SDL_SCANCODE_ESCAPE:
        return true;
    default:
        return false;
    }
}

static unsigned sdl_scancode_to_retrok(SDL_Scancode sc) {
    switch (sc) {
    case SDL_SCANCODE_BACKSPACE:    return 8;
    case SDL_SCANCODE_TAB:          return 9;
    case SDL_SCANCODE_RETURN:       return 13;
    case SDL_SCANCODE_ESCAPE:       return 27;
    case SDL_SCANCODE_SPACE:        return 32;
    case SDL_SCANCODE_COMMA:        return 44;
    case SDL_SCANCODE_MINUS:        return 45;
    case SDL_SCANCODE_PERIOD:       return 46;
    case SDL_SCANCODE_SLASH:        return 47;
    case SDL_SCANCODE_0:            return 48;
    case SDL_SCANCODE_1:            return 49;
    case SDL_SCANCODE_2:            return 50;
    case SDL_SCANCODE_3:            return 51;
    case SDL_SCANCODE_4:            return 52;
    case SDL_SCANCODE_5:            return 53;
    case SDL_SCANCODE_6:            return 54;
    case SDL_SCANCODE_7:            return 55;
    case SDL_SCANCODE_8:            return 56;
    case SDL_SCANCODE_9:            return 57;
    case SDL_SCANCODE_SEMICOLON:    return 59;
    case SDL_SCANCODE_EQUALS:       return 61;
    case SDL_SCANCODE_LEFTBRACKET:  return 91;
    case SDL_SCANCODE_BACKSLASH:    return 92;
    case SDL_SCANCODE_RIGHTBRACKET: return 93;
    case SDL_SCANCODE_GRAVE:        return 96;
    case SDL_SCANCODE_A:            return 97;
    case SDL_SCANCODE_B:            return 98;
    case SDL_SCANCODE_C:            return 99;
    case SDL_SCANCODE_D:            return 100;
    case SDL_SCANCODE_E:            return 101;
    case SDL_SCANCODE_F:            return 102;
    case SDL_SCANCODE_G:            return 103;
    case SDL_SCANCODE_H:            return 104;
    case SDL_SCANCODE_I:            return 105;
    case SDL_SCANCODE_J:            return 106;
    case SDL_SCANCODE_K:            return 107;
    case SDL_SCANCODE_L:            return 108;
    case SDL_SCANCODE_M:            return 109;
    case SDL_SCANCODE_N:            return 110;
    case SDL_SCANCODE_O:            return 111;
    case SDL_SCANCODE_P:            return 112;
    case SDL_SCANCODE_Q:            return 113;
    case SDL_SCANCODE_R:            return 114;
    case SDL_SCANCODE_S:            return 115;
    case SDL_SCANCODE_T:            return 116;
    case SDL_SCANCODE_U:            return 117;
    case SDL_SCANCODE_V:            return 118;
    case SDL_SCANCODE_W:            return 119;
    case SDL_SCANCODE_X:            return 120;
    case SDL_SCANCODE_Y:            return 121;
    case SDL_SCANCODE_Z:            return 122;
    case SDL_SCANCODE_DELETE:       return 127;
    case SDL_SCANCODE_KP_0:         return 256;
    case SDL_SCANCODE_KP_1:         return 257;
    case SDL_SCANCODE_KP_2:         return 258;
    case SDL_SCANCODE_KP_3:         return 259;
    case SDL_SCANCODE_KP_4:         return 260;
    case SDL_SCANCODE_KP_5:         return 261;
    case SDL_SCANCODE_KP_6:         return 262;
    case SDL_SCANCODE_KP_7:         return 263;
    case SDL_SCANCODE_KP_8:         return 264;
    case SDL_SCANCODE_KP_9:         return 265;
    case SDL_SCANCODE_KP_PERIOD:    return 266;
    case SDL_SCANCODE_KP_DIVIDE:    return 267;
    case SDL_SCANCODE_KP_MULTIPLY:  return 268;
    case SDL_SCANCODE_KP_MINUS:     return 269;
    case SDL_SCANCODE_KP_PLUS:      return 270;
    case SDL_SCANCODE_KP_ENTER:     return 271;
    case SDL_SCANCODE_KP_EQUALS:    return 272;
    case SDL_SCANCODE_UP:           return 273;
    case SDL_SCANCODE_DOWN:         return 274;
    case SDL_SCANCODE_RIGHT:        return 275;
    case SDL_SCANCODE_LEFT:         return 276;
    case SDL_SCANCODE_INSERT:       return 277;
    case SDL_SCANCODE_HOME:         return 278;
    case SDL_SCANCODE_END:          return 279;
    case SDL_SCANCODE_PAGEUP:       return 280;
    case SDL_SCANCODE_PAGEDOWN:     return 281;
    case SDL_SCANCODE_F1:           return 282;
    case SDL_SCANCODE_F2:           return 283;
    case SDL_SCANCODE_F3:           return 284;
    case SDL_SCANCODE_F4:           return 285;
    case SDL_SCANCODE_F5:           return 286;
    case SDL_SCANCODE_F6:           return 287;
    case SDL_SCANCODE_F7:           return 288;
    case SDL_SCANCODE_F8:           return 289;
    case SDL_SCANCODE_F9:           return 290;
    case SDL_SCANCODE_F10:          return 291;
    case SDL_SCANCODE_F11:          return 292;
    case SDL_SCANCODE_F12:          return 293;
    case SDL_SCANCODE_NUMLOCKCLEAR: return 300;
    case SDL_SCANCODE_CAPSLOCK:     return 301;
    case SDL_SCANCODE_SCROLLLOCK:   return 302;
    case SDL_SCANCODE_RSHIFT:       return 303;
    case SDL_SCANCODE_LSHIFT:       return 304;
    case SDL_SCANCODE_RCTRL:        return 305;
    case SDL_SCANCODE_LCTRL:        return 306;
    case SDL_SCANCODE_RALT:         return 307;
    case SDL_SCANCODE_LALT:         return 308;
    case SDL_SCANCODE_LGUI:         return 311;
    case SDL_SCANCODE_RGUI:         return 312;
    default:                        return 0;
    }
}

static uint16_t sdl_mod_to_retrokmod(Uint16 sdl_mod) {
    uint16_t mod = 0;
    if (sdl_mod & KMOD_SHIFT) mod |= 0x01;
    if (sdl_mod & KMOD_CTRL)  mod |= 0x02;
    if (sdl_mod & KMOD_ALT)   mod |= 0x04;
    if (sdl_mod & KMOD_GUI)   mod |= 0x08;
    return mod;
}

static void rr_sdl_dispatch_key_event(const SDL_KeyboardEvent* ev) {
    static std::bitset<SDL_NUM_SCANCODES> forwarded_keys;
    const SDL_Scancode sc = ev->keysym.scancode;
    const bool down = ev->type == SDL_KEYDOWN;

    if (rr_keyboard_text_editing()) {
        if (!down) return;
        const unsigned keycode = sdl_scancode_to_retrok(sc);
        if (!keycode) return;
        RRKeyEvent edit_event = {true, keycode, 0, sdl_mod_to_retrokmod(ev->keysym.mod)};
        rr_keyboard_event(&edit_event);
        return;
    }

    if (down && (!rr_keyboard_has_callback() || is_frontend_hotkey(sc))) return;
    if (!down && !forwarded_keys.test(sc)) return;
    const unsigned keycode = sdl_scancode_to_retrok(sc);
    if (keycode == 0) return;

    RRKeyEvent rr_event;
    rr_event.down = down;
    rr_event.keycode = keycode;
    rr_event.character = 0;
    rr_event.modifiers = sdl_mod_to_retrokmod(ev->keysym.mod);
    rr_keyboard_event(&rr_event);
    forwarded_keys.set(sc, down);
}

static uint32_t decode_first_utf8_codepoint(const char* text) {
    const unsigned char* s = reinterpret_cast<const unsigned char*>(text);
    if (!s[0]) return 0;
    if (s[0] < 0x80) return s[0];
    if ((s[0] & 0xe0) == 0xc0 && s[1])
        return ((s[0] & 0x1f) << 6) | (s[1] & 0x3f);
    if ((s[0] & 0xf0) == 0xe0 && s[1] && s[2])
        return ((s[0] & 0x0f) << 12) | ((s[1] & 0x3f) << 6) | (s[2] & 0x3f);
    if ((s[0] & 0xf8) == 0xf0 && s[1] && s[2] && s[3])
        return ((s[0] & 0x07) << 18) | ((s[1] & 0x3f) << 12) |
               ((s[2] & 0x3f) << 6) | (s[3] & 0x3f);
    return 0;
}

static void rr_sdl_dispatch_text_event(const SDL_TextInputEvent* ev) {
    if (!rr_keyboard_has_callback() && !rr_keyboard_text_editing()) return;
    const uint32_t character = decode_first_utf8_codepoint(ev->text);
    if (!character) return;
    RRKeyEvent event = {true, RETROK_UNKNOWN, character, 0};
    rr_keyboard_event(&event);
    event.down = false;
    rr_keyboard_event(&event);
}
