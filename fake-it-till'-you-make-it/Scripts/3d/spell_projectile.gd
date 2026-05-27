extends Node3D

var velocity_dir: Vector3 = Vector3.FORWARD
var speed: float = 20.0
var damage: int = 10
var lifetime: float = 5.0
var spell_name: String = ""
var display_name: String = ""
var display_color: Color = Color.WHITE
var base_visual_scale: Vector3 = Vector3.ONE

@onready var mesh: MeshInstance3D = $MeshInstance3D
@onready var light: OmniLight3D = $OmniLight3D
@onready var area: Area3D = $Area3D
@onready var collision_shape: CollisionShape3D = $Area3D/CollisionShape3D


func _ready() -> void:
	if mesh:
		base_visual_scale = mesh.scale
	_apply_visual_style()
	_expand_hitbox()
	area.monitoring = true
	area.monitorable = true
	var body_callable: Callable = Callable(self, "_on_area_3d_body_entered")
	var area_callable: Callable = Callable(self, "_on_area_3d_area_entered")
	if not area.body_entered.is_connected(body_callable):
		area.body_entered.connect(body_callable)
	if not area.area_entered.is_connected(area_callable):
		area.area_entered.connect(area_callable)


func configure_projectile(new_spell_name: String, new_display_name: String, new_color: Color, new_speed: float, new_damage: int, direction: Vector3) -> void:
	spell_name = new_spell_name
	display_name = new_display_name
	display_color = new_color
	speed = new_speed
	damage = new_damage
	lifetime = 6.0
	if direction != Vector3.ZERO:
		velocity_dir = direction.normalized()
	_apply_visual_style()


func configure_visuals(new_spell_name: String, new_display_name: String, new_color: Color) -> void:
	spell_name = new_spell_name
	display_name = new_display_name
	display_color = new_color
	_apply_visual_style()


func _physics_process(delta: float) -> void:
	global_position += velocity_dir * speed * delta
	rotate_y(delta * 5.0)
	lifetime -= delta
	if lifetime <= 0.0:
		queue_free()


func _apply_visual_style() -> void:
	if mesh:
		var material: StandardMaterial3D = StandardMaterial3D.new()
		material.albedo_color = display_color
		material.emission_enabled = true
		material.emission = display_color
		material.emission_energy_multiplier = 6.0
		mesh.material_override = material
		mesh.scale = base_visual_scale * 2.4
	if light:
		light.light_color = display_color
		light.light_energy = 5.0
		light.omni_range = 10.0


func _expand_hitbox() -> void:
	if collision_shape == null or collision_shape.shape == null:
		return
	if collision_shape.shape is SphereShape3D:
		var sphere: SphereShape3D = collision_shape.shape as SphereShape3D
		sphere.radius = max(sphere.radius, 0.6)


func _on_area_3d_body_entered(body: Node) -> void:
	if body.has_method("take_damage"):
		_spawn_impact_burst()
		body.take_damage(damage)
		queue_free()


func _on_area_3d_area_entered(hit_area: Area3D) -> void:
	if hit_area == null or not is_instance_valid(hit_area):
		return
	var owner: Node = hit_area.get_parent()
	if owner and owner.has_method("take_damage"):
		_spawn_impact_burst()
		owner.take_damage(damage)
		queue_free()


func _spawn_impact_burst() -> void:
	var root: Node = _get_scene_root()
	if root == null:
		return

	var burst: Node3D = Node3D.new()
	root.add_child(burst)
	burst.global_position = global_position

	var flash_mesh: MeshInstance3D = MeshInstance3D.new()
	var sphere_mesh: SphereMesh = SphereMesh.new()
	sphere_mesh.radius = 0.25
	sphere_mesh.height = 0.5
	flash_mesh.mesh = sphere_mesh

	var flash_material: StandardMaterial3D = StandardMaterial3D.new()
	flash_material.albedo_color = display_color
	flash_material.emission_enabled = true
	flash_material.emission = display_color
	flash_material.emission_energy_multiplier = 6.0
	flash_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	flash_mesh.material_override = flash_material
	burst.add_child(flash_mesh)

	var flash_light: OmniLight3D = OmniLight3D.new()
	flash_light.light_color = display_color
	flash_light.light_energy = 4.0
	flash_light.omni_range = 8.0
	burst.add_child(flash_light)

	var burst_tween: Tween = burst.create_tween()
	burst_tween.set_parallel(true)
	burst_tween.tween_property(flash_mesh, "scale", Vector3.ONE * 3.0, 0.16)
	burst_tween.tween_property(flash_light, "light_energy", 0.0, 0.16)
	burst_tween.chain().tween_callback(burst.queue_free)


func _get_scene_root() -> Node:
	if get_tree().current_scene != null:
		return get_tree().current_scene

	var node: Node = self
	while node.get_parent() != null and node.get_parent() != get_tree().root:
		node = node.get_parent()
	return node
