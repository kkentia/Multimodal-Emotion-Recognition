extends Node

@export var spell_hand: Node3D
@export var camera: Camera3D

var SPELLS: Dictionary = {
	"Fireball": {
		"display_name": "Solar Flare",
		"scene": preload("res://Scences/3d/spells/fireball.tscn"),
		"face": "Angry",
		"voice": "Angry",
		"keyword": "ignite",
		"speed": 80.0,
		"damage": 25,
		"color": Color(1.0, 0.35, 0.1, 1.0),
	},
	"Confusion": {
		"display_name": "Pulse Jammer",
		"scene": preload("res://Scences/3d/spells/confusion.tscn"),
		"face": "Happy",
		"voice": "Angry",
		"keyword": "baffle",
		"speed": 50.0,
		"damage": 10,
		"color": Color(0.85, 0.2, 1.0, 1.0),
	},
	"Healing": {
		"display_name": "Repair Wave",
		"scene": preload("res://Scences/3d/spells/healing.tscn"),
		"face": "Happy",
		"voice": "Happy",
		"keyword": "restore",
		"speed": 0.0,
		"damage": -20,
		"color": Color(0.15, 1.0, 0.8, 1.0),
	},
	"Ice Shard": {
		"display_name": "Cryo Burst",
		"scene": preload("res://Scences/3d/spells/ice_shard.tscn"),
		"face": "Sad",
		"voice": "Fear",
		"keyword": "freeze",
		"speed": 70.0,
		"damage": 20,
		"color": Color(0.45, 0.85, 1.0, 1.0),
	},
	"Lightning": {
		"display_name": "Plasma Arc",
		"scene": preload("res://Scences/3d/spells/lightning.tscn"),
		"face": "Fear",
		"voice": "Angry",
		"keyword": "strike",
		"speed": 150.0,
		"damage": 30,
		"color": Color(1.0, 0.9, 0.2, 1.0),
	},
	"Shadow Drain": {
		"display_name": "Gravity Well",
		"scene": preload("res://Scences/3d/spells/shadow_drain.tscn"),
		"face": "Sad",
		"voice": "Sad",
		"keyword": "drain",
		"speed": 60.0,
		"damage": 15,
		"color": Color(0.2, 0.1, 0.35, 1.0),
	},
}

var held_spell_node: Node3D = null
var held_spell_name: String = ""
const PROJECTILE_SPEED_MULTIPLIER: float = 4.0

func get_spell_names() -> Array[String]:
	var names: Array[String] = []
	for spell_name in SPELLS.keys():
		names.append(String(spell_name))
	return names


func find_matching_spell(face: String, voice: String) -> String:
	for spell_name in SPELLS.keys():
		var spell: Dictionary = SPELLS[spell_name]
		if str(spell.get("face", "")).to_lower() == face.to_lower() and str(spell.get("voice", "")).to_lower() == voice.to_lower():
			return spell_name
	return ""


func find_spell_by_constraints(face: String = "", voice: String = "", keyword: String = "") -> String:
	var matches: Array[String] = []
	for spell_name in SPELLS.keys():
		var spell: Dictionary = SPELLS[spell_name]
		var face_matches: bool = face == "" or str(spell.get("face", "")).to_lower() == face.to_lower()
		var voice_matches: bool = voice == "" or str(spell.get("voice", "")).to_lower() == voice.to_lower()
		var keyword_matches: bool = keyword == "" or str(spell.get("keyword", "")).to_lower() == keyword.to_lower()
		if face_matches and voice_matches and keyword_matches:
			matches.append(String(spell_name))

	return matches[0] if matches.size() == 1 else ""


func get_expected_keyword(spell_name: String) -> String:
	var spell: Dictionary = get_ability_data(spell_name)
	return str(spell.get("keyword", ""))


func get_spellbook_entries() -> Array[Dictionary]:
	var entries: Array[Dictionary] = []
	for spell_name in SPELLS.keys():
		var spell: Dictionary = SPELLS[spell_name]
		entries.append({
			"name": String(spell_name),
			"display_name": get_display_name(String(spell_name)),
			"face": str(spell.get("face", "")),
			"voice": str(spell.get("voice", "")),
			"keyword": str(spell.get("keyword", "")),
		})
	return entries


func update_held_spell(face: String, voice: String) -> String:
	var matching_spell: String = find_matching_spell(face, voice)
	if matching_spell != held_spell_name:
		clear_held_spell()
		if matching_spell != "":
			spawn_held_spell(matching_spell)
	return held_spell_name


func select_spell(spell_name: String) -> String:
	if not SPELLS.has(spell_name):
		return held_spell_name
	if held_spell_name != spell_name:
		clear_held_spell()
		spawn_held_spell(spell_name)
	return held_spell_name


func spawn_held_spell(spell_name: String) -> void:
	var spell: Dictionary = SPELLS[spell_name]
	var scene: PackedScene = spell.get("scene", null) as PackedScene
	if scene == null or spell_hand == null:
		return
	var instance: Node3D = scene.instantiate()
	spell_hand.add_child(instance)
	instance.position = Vector3.ZERO
	_apply_spell_visuals(instance, spell_name)

	held_spell_node = instance
	held_spell_name = spell_name
	print("✨ Holding ability: ", get_display_name(spell_name))


func clear_held_spell() -> void:
	if held_spell_node:
		held_spell_node.queue_free()
	held_spell_node = null
	held_spell_name = ""


func cast_spell(spoken_keyword: String) -> bool:
	if held_spell_name == "":
		return false

	var spell: Dictionary = SPELLS[held_spell_name]
	if spoken_keyword.to_lower() != str(spell.get("keyword", "")):
		return false

	return _launch_current_spell(spell)


func cast_current_ability(manual_override: bool = false) -> bool:
	if held_spell_name == "":
		return false
	var spell: Dictionary = SPELLS[held_spell_name]
	if not manual_override:
		return false
	return _launch_current_spell(spell)


func has_held_spell() -> bool:
	return held_spell_name != ""


func get_held_spell_name() -> String:
	return held_spell_name


func get_display_name(spell_name: String) -> String:
	var spell: Dictionary = SPELLS.get(spell_name, {}) as Dictionary
	return str(spell.get("display_name", spell_name))


func get_ability_data(spell_name: String) -> Dictionary:
	return SPELLS.get(spell_name, {}) as Dictionary


func _launch_current_spell(spell: Dictionary) -> bool:
	if held_spell_name == "":
		return false
	if held_spell_node == null or not is_instance_valid(held_spell_node):
		held_spell_node = null
		held_spell_name = ""
		return false

	var spell_name: String = held_spell_name
	var display_name: String = get_display_name(spell_name)
	var scene_root: Node = _get_scene_root()
	var ui: Node = null
	if scene_root != null:
		ui = scene_root.get_node_or_null("UI") as Node
	if ui and ui.has_method("show_casted_message"):
		ui.show_casted_message(display_name)

	if float(spell.get("speed", 0.0)) == 0.0:
		if scene_root != null and scene_root.has_method("apply_player_heal"):
			scene_root.apply_player_heal(abs(int(spell.get("damage", 0))))
		clear_held_spell()
		return true

	var spell_instance: Node3D = held_spell_node
	held_spell_node = null
	held_spell_name = ""

	if camera == null:
		return false
	if spell_instance == null or not is_instance_valid(spell_instance):
		return false
	var ray_dir: Vector3 = -camera.global_transform.basis.z
	var spell_parent: Node = spell_instance.get_parent()
	if spell_parent == null or not is_instance_valid(spell_parent):
		return false
	spell_parent.remove_child(spell_instance)
	if scene_root == null:
		return false
	scene_root.add_child(spell_instance)
	spell_instance.global_position = camera.global_position + ray_dir * 2.0

	if spell_instance.has_method("configure_projectile"):
		var spell_color: Color = spell.get("color", Color.WHITE)
		var projectile_speed: float = float(spell.get("speed", 20.0)) * PROJECTILE_SPEED_MULTIPLIER
		spell_instance.configure_projectile(
			spell_name,
			display_name,
			spell_color,
			float(spell.get("speed", 20.0)),
			projectile_speed,
			int(spell.get("damage", 10)),
			ray_dir
		)
	else:
		spell_instance.velocity_dir = ray_dir
		spell_instance.speed = float(spell.get("speed", 20.0))
		spell_instance.damage = int(spell.get("damage", 10))
		spell_instance.speed = float(spell.get("speed", 20.0)) * PROJECTILE_SPEED_MULTIPLIER
		spell_instance.lifetime = 6.0

	print("🚀 Launched ", display_name, " in direction ", ray_dir, " at speed ", spell.get("speed", 0.0))
	return true


func _apply_spell_visuals(instance: Node3D, spell_name: String) -> void:
	var spell: Dictionary = SPELLS[spell_name]
	if instance.has_method("configure_visuals"):
		var spell_color: Color = spell.get("color", Color.WHITE)
		instance.configure_visuals(
			spell_name,
			get_display_name(spell_name),
			spell_color
		)


func _get_scene_root() -> Node:
	if get_tree().current_scene != null:
		return get_tree().current_scene

	var node: Node = self
	while node.get_parent() != null and node.get_parent() != get_tree().root:
		node = node.get_parent()
	return node
