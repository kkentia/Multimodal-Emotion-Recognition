extends Node

enum InputMode { AI, MANUAL }

var input_mode: InputMode = InputMode.AI

func is_manual() -> bool:
	return input_mode == InputMode.MANUAL

func is_ai() -> bool:
	return input_mode == InputMode.AI
