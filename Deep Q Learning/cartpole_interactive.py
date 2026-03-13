"""
Interactive CartPole - Manual Control
======================================
- The game window shows a "Press ENTER to start" hint
- Use LEFT / RIGHT arrow keys to balance the pole
- One frame every 2 seconds
- Press ESC or close the window to quit

Requirements:
  pip install gymnasium pygame numpy

Run:
  python cartpole_interactive.py
"""

import gymnasium as gym
import numpy as np
import time
import pygame

step_duration = 0.5


def draw_overlay(screen, lines, font, color=(255, 255, 255)):
    """Draw a semi-transparent overlay with centered lines of text."""
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))
    screen.blit(overlay, (0, 0))

    if isinstance(lines, str):
        lines = [lines]

    line_height = font.get_height() + 8
    total_height = line_height * len(lines)
    start_y = screen.get_height() // 2 - total_height // 2

    for i, line in enumerate(lines):
        text = font.render(line, True, color)
        rect = text.get_rect(center=(screen.get_width() // 2, start_y + i * line_height))
        screen.blit(text, rect)
    pygame.display.flip()


def run_keyboard():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    pygame.init()

    # Initial reset to get a frame
    obs, _ = env.reset()
    frame = env.render()  # numpy RGB array

    h, w, _ = frame.shape
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("CartPole - Manual Control")
    font = pygame.font.SysFont(None, 36)

    # Show initial frame with "Press ENTER to start" overlay
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(surf, (0, 0))
    draw_overlay(screen, "Press ENTER to start", font)

    # Wait for ENTER in the pygame window
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    env.close()
                    pygame.quit()
                    return
                elif event.key == pygame.K_RETURN:
                    waiting = False
        time.sleep(0.01)

    total_reward = 0
    action = 1  # default action (right)

    for step in range(500):
        # Wait 2 seconds, use last pressed key as action
        step_start = time.time()
        while time.time() - step_start < step_duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        pygame.quit()
                        return
                    elif event.key == pygame.K_LEFT:
                        action = 0
                    elif event.key == pygame.K_RIGHT:
                        action = 1
            time.sleep(0.01)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Render frame to pygame window
        frame = env.render()
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            break

    # Show game over screen
    draw_overlay(screen, [
        "Game Over!",
        f"Score: {total_reward:.0f}",
        f"Steps: {step + 1}",
        f"Speed: {step_duration}s/frame",
    ], font)
    print(f"\n  Game over! Score: {total_reward:.0f}, Steps survived: {step + 1}")

    # Keep window open until user closes it or presses ESC
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                env.close()
                pygame.quit()
                return
        time.sleep(0.01)


if __name__ == "__main__":
    run_keyboard()