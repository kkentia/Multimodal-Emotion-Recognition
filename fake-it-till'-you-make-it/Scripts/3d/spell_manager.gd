extends Node

@export var spell_hand: Node3D
@export var camera: Camera3D

# Each spell has its own scene file
var SPELLS = {
	"Fireball": {
		"scene": preload("res://Scences/3d/spells/fireball.tscn"),
		"face": "Angry", "voice": "Angry",
		"keyword": "ignite",
		"speed": 30.0,
		"damage": 25
	},
	"Confusion": {
		"scene": preload("res://Scences/3d/spells/confusion.tscn"),
		"face": "Happy", "voice": "Angry",
		"keyword": "baffle",
		"speed": 20.0,
		"damage": 10
	},
	"Healing": {
		"scene": preload("res://Scences/3d/spells/healing.tscn"),
		"face": "Happy", "voice": "Happy",
		"keyword": "restore",
		"speed": 0.0,
		"damage": -20
	},
	"Ice Shard": {
		"scene": preload("res://Scences/3d/spells/ice_shard.tscn"),
		"face": "Sad", "voice": "Fear",
		"keyword": "freeze",
		"speed": 35.0,
		"damage": 20
	},
	"Lightning": {
		"scene": preload("res://Scences/3d/spells/lightning.tscn"),
		"face": "Fear", "voice": "Angry",
		"keyword": "strike",
		"speed": 60.0,
		"damage": 30
	},
	"Shadow Drain": {
		"scene": preload("res://Scences/3d/spells/shadow_drain.tscn"),
		"face": "Sad", "voice": "Sad",
		"keyword": "drain",
		"speed": 25.0,
		"damage": 15
	}
}

var held_spell_node: Node3D = null
var held_spell_name: String = ""


func update_held_spell(face: String, voice: String):
	var matching_spell = ""
	for spell_name in SPELLS.keys():
		var spell = SPELLS[spell_name]
		if spell.face.to_lower() == face.to_lower() and spell.voice.to_lower() == voice.to_lower():
			matching_spell = spell_name
			break
	
	if matching_spell != held_spell_name:
		clear_held_spell()
		if matching_spell != "":
			spawn_held_spell(matching_spell)


func spawn_held_spell(spell_name: String):
	var spell = SPELLS[spell_name]
	var instance = spell.scene.instantiate()
	spell_hand.add_child(instance)
	instance.position = Vector3.ZERO
	
	held_spell_node = instance
	held_spell_name = spell_name
	print("✨ Holding spell: ", spell_name)


func clear_held_spell():
	if held_spell_node:
		held_spell_node.queue_free()
	held_spell_node = null
	held_spell_name = ""


func cast_spell(spoken_keyword: String):
	if held_spell_name == "":
		return
	
	var spell = SPELLS[held_spell_name]
	if spoken_keyword.to_lower() != spell.keyword:
		return
	
	print("🎯 Casting ", held_spell_name, "!")
	
	# Healing — self-cast, just disappear
	if spell.speed == 0:
		clear_held_spell()
		return
	
	# Launch the spell
	var spell_instance = held_spell_node
	held_spell_node = null
	held_spell_name = ""
	
	var scene_root = get_tree().current_scene
	var global_pos = spell_instance.global_position
	spell_instance.get_parent().remove_child(spell_instance)
	scene_root.add_child(spell_instance)
	spell_instance.global_position = global_pos
	
	spell_instance.set_script(preload("res://Scripts/3d/spell_projectile.gd"))
	spell_instance.velocity_dir = -camera.global_transform.basis.z
	spell_instance.speed = spell.speed
	spell_instance.damage = spell.damage
	spell_instance.lifetime = 5.0
