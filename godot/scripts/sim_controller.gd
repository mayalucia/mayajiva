extends Node3D

## Simulation controller — wires LandscapeResource to BugNode and manages lifecycle.

@onready var bug_node: Node = $Bug
@onready var camera: Camera3D = $Camera3D
@onready var terrain_mesh: MeshInstance3D = $Terrain

var landscape_res: Resource


func _ready() -> void:
	# Generate trimesh collision for terrain so raycasts work
	if terrain_mesh:
		terrain_mesh.create_trimesh_collision()

	# Create the landscape resource
	landscape_res = LandscapeResource.new()
	landscape_res.B0 = 50.0
	landscape_res.declination = 0.0
	landscape_res.inclination_deg = 65.0
	landscape_res.sim_width = 1000.0
	landscape_res.sim_height = 1000.0

	# Wire it to the bug
	if bug_node:
		bug_node.landscape_resource = landscape_res
		bug_node.start()

	# Focus camera on terrain centre at reasonable elevation
	if camera and camera.has_method("set_focus"):
		camera.set_focus(Vector3(0.0, 5000.0, 0.0))
