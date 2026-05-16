extends Control

@onready var shader1 = $Shader1Rect
@onready var shader2 = $Shader2Rect
@onready var shader3 = $Shader3Rect
@onready var shader4 = $Shader4Rect
@onready var fps_label = $FPSLabel

var current_shader_index = 0
var shaders = []

func _ready():
	shaders = [shader1, shader2, shader3, shader4]
	
	shader1.material = ShaderMaterial.new()
	shader1.material.shader = load("res://shader1.gdshader")
	
	shader2.material = ShaderMaterial.new()
	shader2.material.shader = load("res://shader2.gdshader")
	
	shader3.material = ShaderMaterial.new()
	shader3.material.shader = load("res://shader3.gdshader")
	
	shader4.material = ShaderMaterial.new()
	shader4.material.shader = load("res://shader4.gdshader")
	
	_update_visibility()

func _process(_delta):
	fps_label.text = "FPS: %d\nPress SPACE to toggle shaders\nCurrently: Shader %d" % [
		Engine.get_frames_per_second(),
		current_shader_index + 1
	]

func _input(event):
	if event.is_action_pressed("ui_accept"):
		current_shader_index = (current_shader_index + 1) % shaders.size()
		_update_visibility()

func _update_visibility():
	for i in range(shaders.size()):
		shaders[i].visible = (i == current_shader_index)
