       гK"	  Ар╫Abrain.Event:2Й6.Й5и     пLыД	ыoЛр╫A"и╨

h
statePlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
Э
,critic/w1_s/Initializer/random_uniform/shapeConst*
_class
loc:@critic/w1_s*
valueB"      *
dtype0*
_output_shapes
:
П
*critic/w1_s/Initializer/random_uniform/minConst*
_class
loc:@critic/w1_s*
valueB
 *░ю╛*
dtype0*
_output_shapes
: 
П
*critic/w1_s/Initializer/random_uniform/maxConst*
_output_shapes
: *
_class
loc:@critic/w1_s*
valueB
 *░ю>*
dtype0
у
4critic/w1_s/Initializer/random_uniform/RandomUniformRandomUniform,critic/w1_s/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А*

seed *
T0*
_class
loc:@critic/w1_s*
seed2 
╩
*critic/w1_s/Initializer/random_uniform/subSub*critic/w1_s/Initializer/random_uniform/max*critic/w1_s/Initializer/random_uniform/min*
T0*
_class
loc:@critic/w1_s*
_output_shapes
: 
▌
*critic/w1_s/Initializer/random_uniform/mulMul4critic/w1_s/Initializer/random_uniform/RandomUniform*critic/w1_s/Initializer/random_uniform/sub*
T0*
_class
loc:@critic/w1_s*
_output_shapes
:	А
╧
&critic/w1_s/Initializer/random_uniformAdd*critic/w1_s/Initializer/random_uniform/mul*critic/w1_s/Initializer/random_uniform/min*
T0*
_class
loc:@critic/w1_s*
_output_shapes
:	А
б
critic/w1_s
VariableV2*
shared_name *
_class
loc:@critic/w1_s*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
─
critic/w1_s/AssignAssigncritic/w1_s&critic/w1_s/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@critic/w1_s*
validate_shape(*
_output_shapes
:	А
s
critic/w1_s/readIdentitycritic/w1_s*
T0*
_class
loc:@critic/w1_s*
_output_shapes
:	А
Щ
*critic/b1/Initializer/random_uniform/shapeConst*
_class
loc:@critic/b1*
valueB"      *
dtype0*
_output_shapes
:
Л
(critic/b1/Initializer/random_uniform/minConst*
_output_shapes
: *
_class
loc:@critic/b1*
valueB
 *Iv╛*
dtype0
Л
(critic/b1/Initializer/random_uniform/maxConst*
_class
loc:@critic/b1*
valueB
 *Iv>*
dtype0*
_output_shapes
: 
▌
2critic/b1/Initializer/random_uniform/RandomUniformRandomUniform*critic/b1/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А*

seed *
T0*
_class
loc:@critic/b1*
seed2 
┬
(critic/b1/Initializer/random_uniform/subSub(critic/b1/Initializer/random_uniform/max(critic/b1/Initializer/random_uniform/min*
_class
loc:@critic/b1*
_output_shapes
: *
T0
╒
(critic/b1/Initializer/random_uniform/mulMul2critic/b1/Initializer/random_uniform/RandomUniform(critic/b1/Initializer/random_uniform/sub*
T0*
_class
loc:@critic/b1*
_output_shapes
:	А
╟
$critic/b1/Initializer/random_uniformAdd(critic/b1/Initializer/random_uniform/mul(critic/b1/Initializer/random_uniform/min*
T0*
_class
loc:@critic/b1*
_output_shapes
:	А
Э
	critic/b1
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *
_class
loc:@critic/b1
╝
critic/b1/AssignAssign	critic/b1$critic/b1/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
:	А
m
critic/b1/readIdentity	critic/b1*
T0*
_class
loc:@critic/b1*
_output_shapes
:	А
Й
critic/MatMulMatMulstatecritic/w1_s/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
c

critic/addAddcritic/MatMulcritic/b1/read*(
_output_shapes
:         А*
T0
R
critic/ReluRelu
critic/add*
T0*(
_output_shapes
:         А
з
1critic/l2/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@critic/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:
Щ
/critic/l2/kernel/Initializer/random_uniform/minConst*#
_class
loc:@critic/l2/kernel*
valueB
 *╫│▌╜*
dtype0*
_output_shapes
: 
Щ
/critic/l2/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@critic/l2/kernel*
valueB
 *╫│▌=*
dtype0*
_output_shapes
: 
є
9critic/l2/kernel/Initializer/random_uniform/RandomUniformRandomUniform1critic/l2/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
АА*

seed *
T0*#
_class
loc:@critic/l2/kernel*
seed2 
▐
/critic/l2/kernel/Initializer/random_uniform/subSub/critic/l2/kernel/Initializer/random_uniform/max/critic/l2/kernel/Initializer/random_uniform/min*#
_class
loc:@critic/l2/kernel*
_output_shapes
: *
T0
Є
/critic/l2/kernel/Initializer/random_uniform/mulMul9critic/l2/kernel/Initializer/random_uniform/RandomUniform/critic/l2/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АА*
T0*#
_class
loc:@critic/l2/kernel
ф
+critic/l2/kernel/Initializer/random_uniformAdd/critic/l2/kernel/Initializer/random_uniform/mul/critic/l2/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@critic/l2/kernel* 
_output_shapes
:
АА
н
critic/l2/kernel
VariableV2*
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *#
_class
loc:@critic/l2/kernel*
	container 
┘
critic/l2/kernel/AssignAssigncritic/l2/kernel+critic/l2/kernel/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@critic/l2/kernel*
validate_shape(* 
_output_shapes
:
АА
Г
critic/l2/kernel/readIdentitycritic/l2/kernel*
T0*#
_class
loc:@critic/l2/kernel* 
_output_shapes
:
АА
Т
 critic/l2/bias/Initializer/zerosConst*!
_class
loc:@critic/l2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Я
critic/l2/bias
VariableV2*!
_class
loc:@critic/l2/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
├
critic/l2/bias/AssignAssigncritic/l2/bias critic/l2/bias/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*!
_class
loc:@critic/l2/bias*
validate_shape(
x
critic/l2/bias/readIdentitycritic/l2/bias*
_output_shapes	
:А*
T0*!
_class
loc:@critic/l2/bias
Ч
critic/l2/MatMulMatMulcritic/Relucritic/l2/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         А*
transpose_a( 
Н
critic/l2/BiasAddBiasAddcritic/l2/MatMulcritic/l2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
\
critic/l2/ReluRelucritic/l2/BiasAdd*
T0*(
_output_shapes
:         А
з
1critic/l3/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*#
_class
loc:@critic/l3/kernel*
valueB"   А   *
dtype0
Щ
/critic/l3/kernel/Initializer/random_uniform/minConst*#
_class
loc:@critic/l3/kernel*
valueB
 *   ╛*
dtype0*
_output_shapes
: 
Щ
/critic/l3/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *#
_class
loc:@critic/l3/kernel*
valueB
 *   >*
dtype0
є
9critic/l3/kernel/Initializer/random_uniform/RandomUniformRandomUniform1critic/l3/kernel/Initializer/random_uniform/shape*
T0*#
_class
loc:@critic/l3/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed 
▐
/critic/l3/kernel/Initializer/random_uniform/subSub/critic/l3/kernel/Initializer/random_uniform/max/critic/l3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@critic/l3/kernel
Є
/critic/l3/kernel/Initializer/random_uniform/mulMul9critic/l3/kernel/Initializer/random_uniform/RandomUniform/critic/l3/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:
АА
ф
+critic/l3/kernel/Initializer/random_uniformAdd/critic/l3/kernel/Initializer/random_uniform/mul/critic/l3/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:
АА
н
critic/l3/kernel
VariableV2*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *#
_class
loc:@critic/l3/kernel
┘
critic/l3/kernel/AssignAssigncritic/l3/kernel+critic/l3/kernel/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@critic/l3/kernel*
validate_shape(* 
_output_shapes
:
АА
Г
critic/l3/kernel/readIdentitycritic/l3/kernel*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:
АА
Т
 critic/l3/bias/Initializer/zerosConst*!
_class
loc:@critic/l3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Я
critic/l3/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *!
_class
loc:@critic/l3/bias*
	container *
shape:А
├
critic/l3/bias/AssignAssigncritic/l3/bias critic/l3/bias/Initializer/zeros*!
_class
loc:@critic/l3/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
x
critic/l3/bias/readIdentitycritic/l3/bias*!
_class
loc:@critic/l3/bias*
_output_shapes	
:А*
T0
Ъ
critic/l3/MatMulMatMulcritic/l2/Relucritic/l3/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
Н
critic/l3/BiasAddBiasAddcritic/l3/MatMulcritic/l3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
\
critic/l3/ReluRelucritic/l3/BiasAdd*(
_output_shapes
:         А*
T0
н
4critic/dense/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@critic/dense/kernel*
valueB"А      *
dtype0*
_output_shapes
:
Я
2critic/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *&
_class
loc:@critic/dense/kernel*
valueB
 *n╫\╛
Я
2critic/dense/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@critic/dense/kernel*
valueB
 *n╫\>*
dtype0*
_output_shapes
: 
√
<critic/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform4critic/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А*

seed *
T0*&
_class
loc:@critic/dense/kernel*
seed2 
ъ
2critic/dense/kernel/Initializer/random_uniform/subSub2critic/dense/kernel/Initializer/random_uniform/max2critic/dense/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
: 
¤
2critic/dense/kernel/Initializer/random_uniform/mulMul<critic/dense/kernel/Initializer/random_uniform/RandomUniform2critic/dense/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	А
я
.critic/dense/kernel/Initializer/random_uniformAdd2critic/dense/kernel/Initializer/random_uniform/mul2critic/dense/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	А
▒
critic/dense/kernel
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *&
_class
loc:@critic/dense/kernel*
	container *
shape:	А
ф
critic/dense/kernel/AssignAssigncritic/dense/kernel.critic/dense/kernel/Initializer/random_uniform*
T0*&
_class
loc:@critic/dense/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(
Л
critic/dense/kernel/readIdentitycritic/dense/kernel*
_output_shapes
:	А*
T0*&
_class
loc:@critic/dense/kernel
Ц
#critic/dense/bias/Initializer/zerosConst*$
_class
loc:@critic/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
г
critic/dense/bias
VariableV2*$
_class
loc:@critic/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
╬
critic/dense/bias/AssignAssigncritic/dense/bias#critic/dense/bias/Initializer/zeros*$
_class
loc:@critic/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
А
critic/dense/bias/readIdentitycritic/dense/bias*
T0*$
_class
loc:@critic/dense/bias*
_output_shapes
:
Я
critic/dense/MatMulMatMulcritic/l3/Relucritic/dense/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
Х
critic/dense/BiasAddBiasAddcritic/dense/MatMulcritic/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
v
critic/discounted_rPlaceholder*'
_output_shapes
:         *
shape:         *
dtype0
n

critic/subSubcritic/discounted_rcritic/dense/BiasAdd*
T0*'
_output_shapes
:         
U
critic/SquareSquare
critic/sub*
T0*'
_output_shapes
:         
]
critic/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
n
critic/MeanMeancritic/Squarecritic/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Y
critic/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
_
critic/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  А?*
dtype0
Д
critic/gradients/FillFillcritic/gradients/Shapecritic/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
А
/critic/gradients/critic/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
│
)critic/gradients/critic/Mean_grad/ReshapeReshapecritic/gradients/Fill/critic/gradients/critic/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
t
'critic/gradients/critic/Mean_grad/ShapeShapecritic/Square*
T0*
out_type0*
_output_shapes
:
╞
&critic/gradients/critic/Mean_grad/TileTile)critic/gradients/critic/Mean_grad/Reshape'critic/gradients/critic/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:         
v
)critic/gradients/critic/Mean_grad/Shape_1Shapecritic/Square*
T0*
out_type0*
_output_shapes
:
l
)critic/gradients/critic/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
q
'critic/gradients/critic/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
└
&critic/gradients/critic/Mean_grad/ProdProd)critic/gradients/critic/Mean_grad/Shape_1'critic/gradients/critic/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
s
)critic/gradients/critic/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
─
(critic/gradients/critic/Mean_grad/Prod_1Prod)critic/gradients/critic/Mean_grad/Shape_2)critic/gradients/critic/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
m
+critic/gradients/critic/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
м
)critic/gradients/critic/Mean_grad/MaximumMaximum(critic/gradients/critic/Mean_grad/Prod_1+critic/gradients/critic/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
к
*critic/gradients/critic/Mean_grad/floordivFloorDiv&critic/gradients/critic/Mean_grad/Prod)critic/gradients/critic/Mean_grad/Maximum*
_output_shapes
: *
T0
К
&critic/gradients/critic/Mean_grad/CastCast*critic/gradients/critic/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
╢
)critic/gradients/critic/Mean_grad/truedivRealDiv&critic/gradients/critic/Mean_grad/Tile&critic/gradients/critic/Mean_grad/Cast*'
_output_shapes
:         *
T0
Ъ
)critic/gradients/critic/Square_grad/ConstConst*^critic/gradients/critic/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
Ч
'critic/gradients/critic/Square_grad/MulMul
critic/sub)critic/gradients/critic/Square_grad/Const*
T0*'
_output_shapes
:         
╢
)critic/gradients/critic/Square_grad/Mul_1Mul)critic/gradients/critic/Mean_grad/truediv'critic/gradients/critic/Square_grad/Mul*'
_output_shapes
:         *
T0
y
&critic/gradients/critic/sub_grad/ShapeShapecritic/discounted_r*
T0*
out_type0*
_output_shapes
:
|
(critic/gradients/critic/sub_grad/Shape_1Shapecritic/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
▐
6critic/gradients/critic/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&critic/gradients/critic/sub_grad/Shape(critic/gradients/critic/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╬
$critic/gradients/critic/sub_grad/SumSum)critic/gradients/critic/Square_grad/Mul_16critic/gradients/critic/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┴
(critic/gradients/critic/sub_grad/ReshapeReshape$critic/gradients/critic/sub_grad/Sum&critic/gradients/critic/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
╥
&critic/gradients/critic/sub_grad/Sum_1Sum)critic/gradients/critic/Square_grad/Mul_18critic/gradients/critic/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
v
$critic/gradients/critic/sub_grad/NegNeg&critic/gradients/critic/sub_grad/Sum_1*
T0*
_output_shapes
:
┼
*critic/gradients/critic/sub_grad/Reshape_1Reshape$critic/gradients/critic/sub_grad/Neg(critic/gradients/critic/sub_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
С
1critic/gradients/critic/sub_grad/tuple/group_depsNoOp)^critic/gradients/critic/sub_grad/Reshape+^critic/gradients/critic/sub_grad/Reshape_1
Т
9critic/gradients/critic/sub_grad/tuple/control_dependencyIdentity(critic/gradients/critic/sub_grad/Reshape2^critic/gradients/critic/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@critic/gradients/critic/sub_grad/Reshape*'
_output_shapes
:         
Ш
;critic/gradients/critic/sub_grad/tuple/control_dependency_1Identity*critic/gradients/critic/sub_grad/Reshape_12^critic/gradients/critic/sub_grad/tuple/group_deps*=
_class3
1/loc:@critic/gradients/critic/sub_grad/Reshape_1*'
_output_shapes
:         *
T0
╛
6critic/gradients/critic/dense/BiasAdd_grad/BiasAddGradBiasAddGrad;critic/gradients/critic/sub_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0
║
;critic/gradients/critic/dense/BiasAdd_grad/tuple/group_depsNoOp7^critic/gradients/critic/dense/BiasAdd_grad/BiasAddGrad<^critic/gradients/critic/sub_grad/tuple/control_dependency_1
╗
Ccritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependencyIdentity;critic/gradients/critic/sub_grad/tuple/control_dependency_1<^critic/gradients/critic/dense/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@critic/gradients/critic/sub_grad/Reshape_1*'
_output_shapes
:         
╖
Ecritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependency_1Identity6critic/gradients/critic/dense/BiasAdd_grad/BiasAddGrad<^critic/gradients/critic/dense/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@critic/gradients/critic/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Є
0critic/gradients/critic/dense/MatMul_grad/MatMulMatMulCcritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependencycritic/dense/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b(*
T0
с
2critic/gradients/critic/dense/MatMul_grad/MatMul_1MatMulcritic/l3/ReluCcritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	А*
transpose_a(*
transpose_b( *
T0
к
:critic/gradients/critic/dense/MatMul_grad/tuple/group_depsNoOp1^critic/gradients/critic/dense/MatMul_grad/MatMul3^critic/gradients/critic/dense/MatMul_grad/MatMul_1
╡
Bcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependencyIdentity0critic/gradients/critic/dense/MatMul_grad/MatMul;^critic/gradients/critic/dense/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@critic/gradients/critic/dense/MatMul_grad/MatMul*(
_output_shapes
:         А
▓
Dcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependency_1Identity2critic/gradients/critic/dense/MatMul_grad/MatMul_1;^critic/gradients/critic/dense/MatMul_grad/tuple/group_deps*E
_class;
97loc:@critic/gradients/critic/dense/MatMul_grad/MatMul_1*
_output_shapes
:	А*
T0
└
-critic/gradients/critic/l3/Relu_grad/ReluGradReluGradBcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependencycritic/l3/Relu*
T0*(
_output_shapes
:         А
о
3critic/gradients/critic/l3/BiasAdd_grad/BiasAddGradBiasAddGrad-critic/gradients/critic/l3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:А*
T0
ж
8critic/gradients/critic/l3/BiasAdd_grad/tuple/group_depsNoOp4^critic/gradients/critic/l3/BiasAdd_grad/BiasAddGrad.^critic/gradients/critic/l3/Relu_grad/ReluGrad
л
@critic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l3/Relu_grad/ReluGrad9^critic/gradients/critic/l3/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@critic/gradients/critic/l3/Relu_grad/ReluGrad*(
_output_shapes
:         А
м
Bcritic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependency_1Identity3critic/gradients/critic/l3/BiasAdd_grad/BiasAddGrad9^critic/gradients/critic/l3/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@critic/gradients/critic/l3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
щ
-critic/gradients/critic/l3/MatMul_grad/MatMulMatMul@critic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependencycritic/l3/kernel/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b(
▄
/critic/gradients/critic/l3/MatMul_grad/MatMul_1MatMulcritic/l2/Relu@critic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( 
б
7critic/gradients/critic/l3/MatMul_grad/tuple/group_depsNoOp.^critic/gradients/critic/l3/MatMul_grad/MatMul0^critic/gradients/critic/l3/MatMul_grad/MatMul_1
й
?critic/gradients/critic/l3/MatMul_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l3/MatMul_grad/MatMul8^critic/gradients/critic/l3/MatMul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*@
_class6
42loc:@critic/gradients/critic/l3/MatMul_grad/MatMul
з
Acritic/gradients/critic/l3/MatMul_grad/tuple/control_dependency_1Identity/critic/gradients/critic/l3/MatMul_grad/MatMul_18^critic/gradients/critic/l3/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@critic/gradients/critic/l3/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
╜
-critic/gradients/critic/l2/Relu_grad/ReluGradReluGrad?critic/gradients/critic/l3/MatMul_grad/tuple/control_dependencycritic/l2/Relu*(
_output_shapes
:         А*
T0
о
3critic/gradients/critic/l2/BiasAdd_grad/BiasAddGradBiasAddGrad-critic/gradients/critic/l2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
ж
8critic/gradients/critic/l2/BiasAdd_grad/tuple/group_depsNoOp4^critic/gradients/critic/l2/BiasAdd_grad/BiasAddGrad.^critic/gradients/critic/l2/Relu_grad/ReluGrad
л
@critic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l2/Relu_grad/ReluGrad9^critic/gradients/critic/l2/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@critic/gradients/critic/l2/Relu_grad/ReluGrad*(
_output_shapes
:         А
м
Bcritic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependency_1Identity3critic/gradients/critic/l2/BiasAdd_grad/BiasAddGrad9^critic/gradients/critic/l2/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*F
_class<
:8loc:@critic/gradients/critic/l2/BiasAdd_grad/BiasAddGrad
щ
-critic/gradients/critic/l2/MatMul_grad/MatMulMatMul@critic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependencycritic/l2/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b(*
T0
┘
/critic/gradients/critic/l2/MatMul_grad/MatMul_1MatMulcritic/Relu@critic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( 
б
7critic/gradients/critic/l2/MatMul_grad/tuple/group_depsNoOp.^critic/gradients/critic/l2/MatMul_grad/MatMul0^critic/gradients/critic/l2/MatMul_grad/MatMul_1
й
?critic/gradients/critic/l2/MatMul_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l2/MatMul_grad/MatMul8^critic/gradients/critic/l2/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@critic/gradients/critic/l2/MatMul_grad/MatMul*(
_output_shapes
:         А
з
Acritic/gradients/critic/l2/MatMul_grad/tuple/control_dependency_1Identity/critic/gradients/critic/l2/MatMul_grad/MatMul_18^critic/gradients/critic/l2/MatMul_grad/tuple/group_deps* 
_output_shapes
:
АА*
T0*B
_class8
64loc:@critic/gradients/critic/l2/MatMul_grad/MatMul_1
╖
*critic/gradients/critic/Relu_grad/ReluGradReluGrad?critic/gradients/critic/l2/MatMul_grad/tuple/control_dependencycritic/Relu*(
_output_shapes
:         А*
T0
s
&critic/gradients/critic/add_grad/ShapeShapecritic/MatMul*
T0*
out_type0*
_output_shapes
:
y
(critic/gradients/critic/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      
▐
6critic/gradients/critic/add_grad/BroadcastGradientArgsBroadcastGradientArgs&critic/gradients/critic/add_grad/Shape(critic/gradients/critic/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╧
$critic/gradients/critic/add_grad/SumSum*critic/gradients/critic/Relu_grad/ReluGrad6critic/gradients/critic/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
┬
(critic/gradients/critic/add_grad/ReshapeReshape$critic/gradients/critic/add_grad/Sum&critic/gradients/critic/add_grad/Shape*(
_output_shapes
:         А*
T0*
Tshape0
╙
&critic/gradients/critic/add_grad/Sum_1Sum*critic/gradients/critic/Relu_grad/ReluGrad8critic/gradients/critic/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┐
*critic/gradients/critic/add_grad/Reshape_1Reshape&critic/gradients/critic/add_grad/Sum_1(critic/gradients/critic/add_grad/Shape_1*
_output_shapes
:	А*
T0*
Tshape0
С
1critic/gradients/critic/add_grad/tuple/group_depsNoOp)^critic/gradients/critic/add_grad/Reshape+^critic/gradients/critic/add_grad/Reshape_1
У
9critic/gradients/critic/add_grad/tuple/control_dependencyIdentity(critic/gradients/critic/add_grad/Reshape2^critic/gradients/critic/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@critic/gradients/critic/add_grad/Reshape*(
_output_shapes
:         А
Р
;critic/gradients/critic/add_grad/tuple/control_dependency_1Identity*critic/gradients/critic/add_grad/Reshape_12^critic/gradients/critic/add_grad/tuple/group_deps*
_output_shapes
:	А*
T0*=
_class3
1/loc:@critic/gradients/critic/add_grad/Reshape_1
┘
*critic/gradients/critic/MatMul_grad/MatMulMatMul9critic/gradients/critic/add_grad/tuple/control_dependencycritic/w1_s/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b(
╚
,critic/gradients/critic/MatMul_grad/MatMul_1MatMulstate9critic/gradients/critic/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	А*
transpose_a(
Ш
4critic/gradients/critic/MatMul_grad/tuple/group_depsNoOp+^critic/gradients/critic/MatMul_grad/MatMul-^critic/gradients/critic/MatMul_grad/MatMul_1
Ь
<critic/gradients/critic/MatMul_grad/tuple/control_dependencyIdentity*critic/gradients/critic/MatMul_grad/MatMul5^critic/gradients/critic/MatMul_grad/tuple/group_deps*'
_output_shapes
:         *
T0*=
_class3
1/loc:@critic/gradients/critic/MatMul_grad/MatMul
Ъ
>critic/gradients/critic/MatMul_grad/tuple/control_dependency_1Identity,critic/gradients/critic/MatMul_grad/MatMul_15^critic/gradients/critic/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@critic/gradients/critic/MatMul_grad/MatMul_1*
_output_shapes
:	А
Г
 critic/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@critic/b1*
valueB
 *fff?
Ф
critic/beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@critic/b1*
	container 
┴
critic/beta1_power/AssignAssigncritic/beta1_power critic/beta1_power/initial_value*
use_locking(*
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
: 
v
critic/beta1_power/readIdentitycritic/beta1_power*
_output_shapes
: *
T0*
_class
loc:@critic/b1
Г
 critic/beta2_power/initial_valueConst*
_class
loc:@critic/b1*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
Ф
critic/beta2_power
VariableV2*
_output_shapes
: *
shared_name *
_class
loc:@critic/b1*
	container *
shape: *
dtype0
┴
critic/beta2_power/AssignAssigncritic/beta2_power critic/beta2_power/initial_value*
use_locking(*
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
: 
v
critic/beta2_power/readIdentitycritic/beta2_power*
_output_shapes
: *
T0*
_class
loc:@critic/b1
к
9critic/critic/w1_s/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@critic/w1_s*
valueB"      *
dtype0*
_output_shapes
:
Ф
/critic/critic/w1_s/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@critic/w1_s*
valueB
 *    
∙
)critic/critic/w1_s/Adam/Initializer/zerosFill9critic/critic/w1_s/Adam/Initializer/zeros/shape_as_tensor/critic/critic/w1_s/Adam/Initializer/zeros/Const*
_output_shapes
:	А*
T0*
_class
loc:@critic/w1_s*

index_type0
н
critic/critic/w1_s/Adam
VariableV2*
_class
loc:@critic/w1_s*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name 
▀
critic/critic/w1_s/Adam/AssignAssigncritic/critic/w1_s/Adam)critic/critic/w1_s/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@critic/w1_s*
validate_shape(*
_output_shapes
:	А
Л
critic/critic/w1_s/Adam/readIdentitycritic/critic/w1_s/Adam*
_output_shapes
:	А*
T0*
_class
loc:@critic/w1_s
м
;critic/critic/w1_s/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@critic/w1_s*
valueB"      *
dtype0*
_output_shapes
:
Ц
1critic/critic/w1_s/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@critic/w1_s*
valueB
 *    
 
+critic/critic/w1_s/Adam_1/Initializer/zerosFill;critic/critic/w1_s/Adam_1/Initializer/zeros/shape_as_tensor1critic/critic/w1_s/Adam_1/Initializer/zeros/Const*
_output_shapes
:	А*
T0*
_class
loc:@critic/w1_s*

index_type0
п
critic/critic/w1_s/Adam_1
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *
_class
loc:@critic/w1_s*
	container *
shape:	А
х
 critic/critic/w1_s/Adam_1/AssignAssigncritic/critic/w1_s/Adam_1+critic/critic/w1_s/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@critic/w1_s*
validate_shape(*
_output_shapes
:	А
П
critic/critic/w1_s/Adam_1/readIdentitycritic/critic/w1_s/Adam_1*
_output_shapes
:	А*
T0*
_class
loc:@critic/w1_s
Ь
'critic/critic/b1/Adam/Initializer/zerosConst*
_class
loc:@critic/b1*
valueB	А*    *
dtype0*
_output_shapes
:	А
й
critic/critic/b1/Adam
VariableV2*
_class
loc:@critic/b1*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name 
╫
critic/critic/b1/Adam/AssignAssigncritic/critic/b1/Adam'critic/critic/b1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
:	А
Е
critic/critic/b1/Adam/readIdentitycritic/critic/b1/Adam*
T0*
_class
loc:@critic/b1*
_output_shapes
:	А
Ю
)critic/critic/b1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	А*
_class
loc:@critic/b1*
valueB	А*    
л
critic/critic/b1/Adam_1
VariableV2*
_class
loc:@critic/b1*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name 
▌
critic/critic/b1/Adam_1/AssignAssigncritic/critic/b1/Adam_1)critic/critic/b1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
:	А
Й
critic/critic/b1/Adam_1/readIdentitycritic/critic/b1/Adam_1*
T0*
_class
loc:@critic/b1*
_output_shapes
:	А
┤
>critic/critic/l2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@critic/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ю
4critic/critic/l2/kernel/Adam/Initializer/zeros/ConstConst*#
_class
loc:@critic/l2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
О
.critic/critic/l2/kernel/Adam/Initializer/zerosFill>critic/critic/l2/kernel/Adam/Initializer/zeros/shape_as_tensor4critic/critic/l2/kernel/Adam/Initializer/zeros/Const*
T0*#
_class
loc:@critic/l2/kernel*

index_type0* 
_output_shapes
:
АА
╣
critic/critic/l2/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *#
_class
loc:@critic/l2/kernel*
	container *
shape:
АА
Ї
#critic/critic/l2/kernel/Adam/AssignAssigncritic/critic/l2/kernel/Adam.critic/critic/l2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@critic/l2/kernel*
validate_shape(* 
_output_shapes
:
АА
Ы
!critic/critic/l2/kernel/Adam/readIdentitycritic/critic/l2/kernel/Adam*
T0*#
_class
loc:@critic/l2/kernel* 
_output_shapes
:
АА
╢
@critic/critic/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@critic/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:
а
6critic/critic/l2/kernel/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@critic/l2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
0critic/critic/l2/kernel/Adam_1/Initializer/zerosFill@critic/critic/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensor6critic/critic/l2/kernel/Adam_1/Initializer/zeros/Const*
T0*#
_class
loc:@critic/l2/kernel*

index_type0* 
_output_shapes
:
АА
╗
critic/critic/l2/kernel/Adam_1
VariableV2*#
_class
loc:@critic/l2/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name 
·
%critic/critic/l2/kernel/Adam_1/AssignAssigncritic/critic/l2/kernel/Adam_10critic/critic/l2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@critic/l2/kernel*
validate_shape(* 
_output_shapes
:
АА
Я
#critic/critic/l2/kernel/Adam_1/readIdentitycritic/critic/l2/kernel/Adam_1* 
_output_shapes
:
АА*
T0*#
_class
loc:@critic/l2/kernel
Ю
,critic/critic/l2/bias/Adam/Initializer/zerosConst*!
_class
loc:@critic/l2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
л
critic/critic/l2/bias/Adam
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *!
_class
loc:@critic/l2/bias
ч
!critic/critic/l2/bias/Adam/AssignAssigncritic/critic/l2/bias/Adam,critic/critic/l2/bias/Adam/Initializer/zeros*!
_class
loc:@critic/l2/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
Р
critic/critic/l2/bias/Adam/readIdentitycritic/critic/l2/bias/Adam*!
_class
loc:@critic/l2/bias*
_output_shapes	
:А*
T0
а
.critic/critic/l2/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@critic/l2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
н
critic/critic/l2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *!
_class
loc:@critic/l2/bias*
	container *
shape:А
э
#critic/critic/l2/bias/Adam_1/AssignAssigncritic/critic/l2/bias/Adam_1.critic/critic/l2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@critic/l2/bias*
validate_shape(*
_output_shapes	
:А
Ф
!critic/critic/l2/bias/Adam_1/readIdentitycritic/critic/l2/bias/Adam_1*
T0*!
_class
loc:@critic/l2/bias*
_output_shapes	
:А
┤
>critic/critic/l3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@critic/l3/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
Ю
4critic/critic/l3/kernel/Adam/Initializer/zeros/ConstConst*#
_class
loc:@critic/l3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
О
.critic/critic/l3/kernel/Adam/Initializer/zerosFill>critic/critic/l3/kernel/Adam/Initializer/zeros/shape_as_tensor4critic/critic/l3/kernel/Adam/Initializer/zeros/Const*
T0*#
_class
loc:@critic/l3/kernel*

index_type0* 
_output_shapes
:
АА
╣
critic/critic/l3/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *#
_class
loc:@critic/l3/kernel*
	container *
shape:
АА
Ї
#critic/critic/l3/kernel/Adam/AssignAssigncritic/critic/l3/kernel/Adam.critic/critic/l3/kernel/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@critic/l3/kernel*
validate_shape(* 
_output_shapes
:
АА
Ы
!critic/critic/l3/kernel/Adam/readIdentitycritic/critic/l3/kernel/Adam*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:
АА
╢
@critic/critic/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@critic/l3/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
а
6critic/critic/l3/kernel/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@critic/l3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
0critic/critic/l3/kernel/Adam_1/Initializer/zerosFill@critic/critic/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensor6critic/critic/l3/kernel/Adam_1/Initializer/zeros/Const*
T0*#
_class
loc:@critic/l3/kernel*

index_type0* 
_output_shapes
:
АА
╗
critic/critic/l3/kernel/Adam_1
VariableV2*
shared_name *#
_class
loc:@critic/l3/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
·
%critic/critic/l3/kernel/Adam_1/AssignAssigncritic/critic/l3/kernel/Adam_10critic/critic/l3/kernel/Adam_1/Initializer/zeros*#
_class
loc:@critic/l3/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0
Я
#critic/critic/l3/kernel/Adam_1/readIdentitycritic/critic/l3/kernel/Adam_1*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:
АА
Ю
,critic/critic/l3/bias/Adam/Initializer/zerosConst*!
_class
loc:@critic/l3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
л
critic/critic/l3/bias/Adam
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *!
_class
loc:@critic/l3/bias
ч
!critic/critic/l3/bias/Adam/AssignAssigncritic/critic/l3/bias/Adam,critic/critic/l3/bias/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@critic/l3/bias*
validate_shape(*
_output_shapes	
:А
Р
critic/critic/l3/bias/Adam/readIdentitycritic/critic/l3/bias/Adam*
_output_shapes	
:А*
T0*!
_class
loc:@critic/l3/bias
а
.critic/critic/l3/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@critic/l3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
н
critic/critic/l3/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *!
_class
loc:@critic/l3/bias*
	container *
shape:А
э
#critic/critic/l3/bias/Adam_1/AssignAssigncritic/critic/l3/bias/Adam_1.critic/critic/l3/bias/Adam_1/Initializer/zeros*!
_class
loc:@critic/l3/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
Ф
!critic/critic/l3/bias/Adam_1/readIdentitycritic/critic/l3/bias/Adam_1*
T0*!
_class
loc:@critic/l3/bias*
_output_shapes	
:А
░
1critic/critic/dense/kernel/Adam/Initializer/zerosConst*&
_class
loc:@critic/dense/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
╜
critic/critic/dense/kernel/Adam
VariableV2*&
_class
loc:@critic/dense/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name 
 
&critic/critic/dense/kernel/Adam/AssignAssigncritic/critic/dense/kernel/Adam1critic/critic/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@critic/dense/kernel*
validate_shape(*
_output_shapes
:	А
г
$critic/critic/dense/kernel/Adam/readIdentitycritic/critic/dense/kernel/Adam*
_output_shapes
:	А*
T0*&
_class
loc:@critic/dense/kernel
▓
3critic/critic/dense/kernel/Adam_1/Initializer/zerosConst*&
_class
loc:@critic/dense/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
┐
!critic/critic/dense/kernel/Adam_1
VariableV2*
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *&
_class
loc:@critic/dense/kernel*
	container 
Е
(critic/critic/dense/kernel/Adam_1/AssignAssign!critic/critic/dense/kernel/Adam_13critic/critic/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@critic/dense/kernel*
validate_shape(*
_output_shapes
:	А
з
&critic/critic/dense/kernel/Adam_1/readIdentity!critic/critic/dense/kernel/Adam_1*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	А
в
/critic/critic/dense/bias/Adam/Initializer/zerosConst*
_output_shapes
:*$
_class
loc:@critic/dense/bias*
valueB*    *
dtype0
п
critic/critic/dense/bias/Adam
VariableV2*
_output_shapes
:*
shared_name *$
_class
loc:@critic/dense/bias*
	container *
shape:*
dtype0
Є
$critic/critic/dense/bias/Adam/AssignAssigncritic/critic/dense/bias/Adam/critic/critic/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@critic/dense/bias*
validate_shape(*
_output_shapes
:
Ш
"critic/critic/dense/bias/Adam/readIdentitycritic/critic/dense/bias/Adam*$
_class
loc:@critic/dense/bias*
_output_shapes
:*
T0
д
1critic/critic/dense/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@critic/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
▒
critic/critic/dense/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@critic/dense/bias*
	container 
°
&critic/critic/dense/bias/Adam_1/AssignAssigncritic/critic/dense/bias/Adam_11critic/critic/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@critic/dense/bias*
validate_shape(*
_output_shapes
:
Ь
$critic/critic/dense/bias/Adam_1/readIdentitycritic/critic/dense/bias/Adam_1*
T0*$
_class
loc:@critic/dense/bias*
_output_shapes
:
^
critic/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *╖Q9
V
critic/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
V
critic/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w╛?
X
critic/Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
п
(critic/Adam/update_critic/w1_s/ApplyAdam	ApplyAdamcritic/w1_scritic/critic/w1_s/Adamcritic/critic/w1_s/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilon>critic/gradients/critic/MatMul_grad/tuple/control_dependency_1*
_class
loc:@critic/w1_s*
use_nesterov( *
_output_shapes
:	А*
use_locking( *
T0
в
&critic/Adam/update_critic/b1/ApplyAdam	ApplyAdam	critic/b1critic/critic/b1/Adamcritic/critic/b1/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilon;critic/gradients/critic/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	А*
use_locking( *
T0*
_class
loc:@critic/b1
╠
-critic/Adam/update_critic/l2/kernel/ApplyAdam	ApplyAdamcritic/l2/kernelcritic/critic/l2/kernel/Adamcritic/critic/l2/kernel/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonAcritic/gradients/critic/l2/MatMul_grad/tuple/control_dependency_1*#
_class
loc:@critic/l2/kernel*
use_nesterov( * 
_output_shapes
:
АА*
use_locking( *
T0
╛
+critic/Adam/update_critic/l2/bias/ApplyAdam	ApplyAdamcritic/l2/biascritic/critic/l2/bias/Adamcritic/critic/l2/bias/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonBcritic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@critic/l2/bias*
use_nesterov( *
_output_shapes	
:А
╠
-critic/Adam/update_critic/l3/kernel/ApplyAdam	ApplyAdamcritic/l3/kernelcritic/critic/l3/kernel/Adamcritic/critic/l3/kernel/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonAcritic/gradients/critic/l3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@critic/l3/kernel*
use_nesterov( * 
_output_shapes
:
АА
╛
+critic/Adam/update_critic/l3/bias/ApplyAdam	ApplyAdamcritic/l3/biascritic/critic/l3/bias/Adamcritic/critic/l3/bias/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonBcritic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@critic/l3/bias*
use_nesterov( *
_output_shapes	
:А
▌
0critic/Adam/update_critic/dense/kernel/ApplyAdam	ApplyAdamcritic/dense/kernelcritic/critic/dense/kernel/Adam!critic/critic/dense/kernel/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonDcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependency_1*&
_class
loc:@critic/dense/kernel*
use_nesterov( *
_output_shapes
:	А*
use_locking( *
T0
╧
.critic/Adam/update_critic/dense/bias/ApplyAdam	ApplyAdamcritic/dense/biascritic/critic/dense/bias/Adamcritic/critic/dense/bias/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonEcritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@critic/dense/bias*
use_nesterov( *
_output_shapes
:
ї
critic/Adam/mulMulcritic/beta1_power/readcritic/Adam/beta1'^critic/Adam/update_critic/b1/ApplyAdam/^critic/Adam/update_critic/dense/bias/ApplyAdam1^critic/Adam/update_critic/dense/kernel/ApplyAdam,^critic/Adam/update_critic/l2/bias/ApplyAdam.^critic/Adam/update_critic/l2/kernel/ApplyAdam,^critic/Adam/update_critic/l3/bias/ApplyAdam.^critic/Adam/update_critic/l3/kernel/ApplyAdam)^critic/Adam/update_critic/w1_s/ApplyAdam*
T0*
_class
loc:@critic/b1*
_output_shapes
: 
й
critic/Adam/AssignAssigncritic/beta1_powercritic/Adam/mul*
use_locking( *
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
: 
ў
critic/Adam/mul_1Mulcritic/beta2_power/readcritic/Adam/beta2'^critic/Adam/update_critic/b1/ApplyAdam/^critic/Adam/update_critic/dense/bias/ApplyAdam1^critic/Adam/update_critic/dense/kernel/ApplyAdam,^critic/Adam/update_critic/l2/bias/ApplyAdam.^critic/Adam/update_critic/l2/kernel/ApplyAdam,^critic/Adam/update_critic/l3/bias/ApplyAdam.^critic/Adam/update_critic/l3/kernel/ApplyAdam)^critic/Adam/update_critic/w1_s/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@critic/b1
н
critic/Adam/Assign_1Assigncritic/beta2_powercritic/Adam/mul_1*
use_locking( *
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
: 
│
critic/AdamNoOp^critic/Adam/Assign^critic/Adam/Assign_1'^critic/Adam/update_critic/b1/ApplyAdam/^critic/Adam/update_critic/dense/bias/ApplyAdam1^critic/Adam/update_critic/dense/kernel/ApplyAdam,^critic/Adam/update_critic/l2/bias/ApplyAdam.^critic/Adam/update_critic/l2/kernel/ApplyAdam,^critic/Adam/update_critic/l3/bias/ApplyAdam.^critic/Adam/update_critic/l3/kernel/ApplyAdam)^critic/Adam/update_critic/w1_s/ApplyAdam
Я
-pi/l1/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/l1/kernel*
valueB"      *
dtype0*
_output_shapes
:
С
+pi/l1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
_class
loc:@pi/l1/kernel*
valueB
 *░ю╛
С
+pi/l1/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/l1/kernel*
valueB
 *░ю>*
dtype0*
_output_shapes
: 
ц
5pi/l1/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l1/kernel/Initializer/random_uniform/shape*
_class
loc:@pi/l1/kernel*
seed2 *
dtype0*
_output_shapes
:	А*

seed *
T0
╬
+pi/l1/kernel/Initializer/random_uniform/subSub+pi/l1/kernel/Initializer/random_uniform/max+pi/l1/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l1/kernel*
_output_shapes
: 
с
+pi/l1/kernel/Initializer/random_uniform/mulMul5pi/l1/kernel/Initializer/random_uniform/RandomUniform+pi/l1/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@pi/l1/kernel*
_output_shapes
:	А
╙
'pi/l1/kernel/Initializer/random_uniformAdd+pi/l1/kernel/Initializer/random_uniform/mul+pi/l1/kernel/Initializer/random_uniform/min*
_class
loc:@pi/l1/kernel*
_output_shapes
:	А*
T0
г
pi/l1/kernel
VariableV2*
_class
loc:@pi/l1/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name 
╚
pi/l1/kernel/AssignAssignpi/l1/kernel'pi/l1/kernel/Initializer/random_uniform*
_class
loc:@pi/l1/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0
v
pi/l1/kernel/readIdentitypi/l1/kernel*
_class
loc:@pi/l1/kernel*
_output_shapes
:	А*
T0
К
pi/l1/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*
_class
loc:@pi/l1/bias*
valueBА*    
Ч

pi/l1/bias
VariableV2*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l1/bias*
	container *
shape:А*
dtype0
│
pi/l1/bias/AssignAssign
pi/l1/biaspi/l1/bias/Initializer/zeros*
_class
loc:@pi/l1/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
l
pi/l1/bias/readIdentity
pi/l1/bias*
T0*
_class
loc:@pi/l1/bias*
_output_shapes	
:А
Й
pi/l1/MatMulMatMulstatepi/l1/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
Б
pi/l1/BiasAddBiasAddpi/l1/MatMulpi/l1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
T

pi/l1/ReluRelupi/l1/BiasAdd*
T0*(
_output_shapes
:         А
Я
-pi/l2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@pi/l2/kernel*
valueB"      
С
+pi/l2/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/l2/kernel*
valueB
 *╫│▌╜*
dtype0*
_output_shapes
: 
С
+pi/l2/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/l2/kernel*
valueB
 *╫│▌=*
dtype0*
_output_shapes
: 
ч
5pi/l2/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l2/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@pi/l2/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed 
╬
+pi/l2/kernel/Initializer/random_uniform/subSub+pi/l2/kernel/Initializer/random_uniform/max+pi/l2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@pi/l2/kernel
т
+pi/l2/kernel/Initializer/random_uniform/mulMul5pi/l2/kernel/Initializer/random_uniform/RandomUniform+pi/l2/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@pi/l2/kernel* 
_output_shapes
:
АА
╘
'pi/l2/kernel/Initializer/random_uniformAdd+pi/l2/kernel/Initializer/random_uniform/mul+pi/l2/kernel/Initializer/random_uniform/min* 
_output_shapes
:
АА*
T0*
_class
loc:@pi/l2/kernel
е
pi/l2/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *
_class
loc:@pi/l2/kernel*
	container *
shape:
АА
╔
pi/l2/kernel/AssignAssignpi/l2/kernel'pi/l2/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@pi/l2/kernel*
validate_shape(* 
_output_shapes
:
АА
w
pi/l2/kernel/readIdentitypi/l2/kernel*
T0*
_class
loc:@pi/l2/kernel* 
_output_shapes
:
АА
К
pi/l2/bias/Initializer/zerosConst*
_class
loc:@pi/l2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Ч

pi/l2/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l2/bias*
	container *
shape:А
│
pi/l2/bias/AssignAssign
pi/l2/biaspi/l2/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@pi/l2/bias
l
pi/l2/bias/readIdentity
pi/l2/bias*
T0*
_class
loc:@pi/l2/bias*
_output_shapes	
:А
О
pi/l2/MatMulMatMul
pi/l1/Relupi/l2/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
Б
pi/l2/BiasAddBiasAddpi/l2/MatMulpi/l2/bias/read*
data_formatNHWC*(
_output_shapes
:         А*
T0
T

pi/l2/ReluRelupi/l2/BiasAdd*
T0*(
_output_shapes
:         А
Я
-pi/l3/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:
С
+pi/l3/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/l3/kernel*
valueB
 *╫│▌╜*
dtype0*
_output_shapes
: 
С
+pi/l3/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
_class
loc:@pi/l3/kernel*
valueB
 *╫│▌=*
dtype0
ч
5pi/l3/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l3/kernel/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@pi/l3/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА
╬
+pi/l3/kernel/Initializer/random_uniform/subSub+pi/l3/kernel/Initializer/random_uniform/max+pi/l3/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l3/kernel*
_output_shapes
: 
т
+pi/l3/kernel/Initializer/random_uniform/mulMul5pi/l3/kernel/Initializer/random_uniform/RandomUniform+pi/l3/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:
АА
╘
'pi/l3/kernel/Initializer/random_uniformAdd+pi/l3/kernel/Initializer/random_uniform/mul+pi/l3/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:
АА
е
pi/l3/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *
_class
loc:@pi/l3/kernel*
	container *
shape:
АА
╔
pi/l3/kernel/AssignAssignpi/l3/kernel'pi/l3/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*
_class
loc:@pi/l3/kernel
w
pi/l3/kernel/readIdentitypi/l3/kernel*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:
АА
К
pi/l3/bias/Initializer/zerosConst*
_class
loc:@pi/l3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Ч

pi/l3/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l3/bias*
	container *
shape:А
│
pi/l3/bias/AssignAssign
pi/l3/biaspi/l3/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l3/bias*
validate_shape(*
_output_shapes	
:А
l
pi/l3/bias/readIdentity
pi/l3/bias*
T0*
_class
loc:@pi/l3/bias*
_output_shapes	
:А
О
pi/l3/MatMulMatMul
pi/l2/Relupi/l3/kernel/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b( 
Б
pi/l3/BiasAddBiasAddpi/l3/MatMulpi/l3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
T

pi/l3/ReluRelupi/l3/BiasAdd*(
_output_shapes
:         А*
T0
Я
-pi/l4/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/l4/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
С
+pi/l4/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/l4/kernel*
valueB
 *   ╛*
dtype0*
_output_shapes
: 
С
+pi/l4/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/l4/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
ч
5pi/l4/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l4/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
АА*

seed *
T0*
_class
loc:@pi/l4/kernel*
seed2 *
dtype0
╬
+pi/l4/kernel/Initializer/random_uniform/subSub+pi/l4/kernel/Initializer/random_uniform/max+pi/l4/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l4/kernel*
_output_shapes
: 
т
+pi/l4/kernel/Initializer/random_uniform/mulMul5pi/l4/kernel/Initializer/random_uniform/RandomUniform+pi/l4/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@pi/l4/kernel* 
_output_shapes
:
АА
╘
'pi/l4/kernel/Initializer/random_uniformAdd+pi/l4/kernel/Initializer/random_uniform/mul+pi/l4/kernel/Initializer/random_uniform/min*
_class
loc:@pi/l4/kernel* 
_output_shapes
:
АА*
T0
е
pi/l4/kernel
VariableV2*
shared_name *
_class
loc:@pi/l4/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
╔
pi/l4/kernel/AssignAssignpi/l4/kernel'pi/l4/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*
_class
loc:@pi/l4/kernel
w
pi/l4/kernel/readIdentitypi/l4/kernel* 
_output_shapes
:
АА*
T0*
_class
loc:@pi/l4/kernel
К
pi/l4/bias/Initializer/zerosConst*
_class
loc:@pi/l4/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Ч

pi/l4/bias
VariableV2*
shared_name *
_class
loc:@pi/l4/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
│
pi/l4/bias/AssignAssign
pi/l4/biaspi/l4/bias/Initializer/zeros*
_class
loc:@pi/l4/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
l
pi/l4/bias/readIdentity
pi/l4/bias*
_output_shapes	
:А*
T0*
_class
loc:@pi/l4/bias
О
pi/l4/MatMulMatMul
pi/l3/Relupi/l4/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         А*
transpose_a( 
Б
pi/l4/BiasAddBiasAddpi/l4/MatMulpi/l4/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
T

pi/l4/ReluRelupi/l4/BiasAdd*(
_output_shapes
:         А*
T0
Э
,pi/a/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/a/kernel*
valueB"А      *
dtype0*
_output_shapes
:
П
*pi/a/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/a/kernel*
valueB
 *JQZ╛*
dtype0*
_output_shapes
: 
П
*pi/a/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/a/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
у
4pi/a/kernel/Initializer/random_uniform/RandomUniformRandomUniform,pi/a/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А*

seed *
T0*
_class
loc:@pi/a/kernel*
seed2 
╩
*pi/a/kernel/Initializer/random_uniform/subSub*pi/a/kernel/Initializer/random_uniform/max*pi/a/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@pi/a/kernel
▌
*pi/a/kernel/Initializer/random_uniform/mulMul4pi/a/kernel/Initializer/random_uniform/RandomUniform*pi/a/kernel/Initializer/random_uniform/sub*
_output_shapes
:	А*
T0*
_class
loc:@pi/a/kernel
╧
&pi/a/kernel/Initializer/random_uniformAdd*pi/a/kernel/Initializer/random_uniform/mul*pi/a/kernel/Initializer/random_uniform/min*
_class
loc:@pi/a/kernel*
_output_shapes
:	А*
T0
б
pi/a/kernel
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *
_class
loc:@pi/a/kernel*
	container *
shape:	А
─
pi/a/kernel/AssignAssignpi/a/kernel&pi/a/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*
_class
loc:@pi/a/kernel
s
pi/a/kernel/readIdentitypi/a/kernel*
T0*
_class
loc:@pi/a/kernel*
_output_shapes
:	А
Ж
pi/a/bias/Initializer/zerosConst*
_class
loc:@pi/a/bias*
valueB*    *
dtype0*
_output_shapes
:
У
	pi/a/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@pi/a/bias
о
pi/a/bias/AssignAssign	pi/a/biaspi/a/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
:
h
pi/a/bias/readIdentity	pi/a/bias*
T0*
_class
loc:@pi/a/bias*
_output_shapes
:
Л
pi/a/MatMulMatMul
pi/l4/Relupi/a/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
}
pi/a/BiasAddBiasAddpi/a/MatMulpi/a/bias/read*'
_output_shapes
:         *
T0*
data_formatNHWC
Q
	pi/a/TanhTanhpi/a/BiasAdd*
T0*'
_output_shapes
:         
е
0pi/dense/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@pi/dense/kernel*
valueB"А      *
dtype0*
_output_shapes
:
Ч
.pi/dense/kernel/Initializer/random_uniform/minConst*"
_class
loc:@pi/dense/kernel*
valueB
 *JQZ╛*
dtype0*
_output_shapes
: 
Ч
.pi/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@pi/dense/kernel*
valueB
 *JQZ>*
dtype0
я
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@pi/dense/kernel*
seed2 *
dtype0*
_output_shapes
:	А*

seed 
┌
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@pi/dense/kernel*
_output_shapes
: *
T0
э
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	А
▀
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	А
й
pi/dense/kernel
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	А
╘
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	А

pi/dense/kernel/readIdentitypi/dense/kernel*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	А
О
pi/dense/bias/Initializer/zerosConst*
_output_shapes
:* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0
Ы
pi/dense/bias
VariableV2* 
_class
loc:@pi/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
╛
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
t
pi/dense/bias/readIdentitypi/dense/bias* 
_class
loc:@pi/dense/bias*
_output_shapes
:*
T0
У
pi/dense/MatMulMatMul
pi/l4/Relupi/dense/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
Й
pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
a
pi/dense/SoftplusSoftpluspi/dense/BiasAdd*
T0*'
_output_shapes
:         
V
pi/Normal/locIdentity	pi/a/Tanh*'
_output_shapes
:         *
T0
`
pi/Normal/scaleIdentitypi/dense/Softplus*
T0*'
_output_shapes
:         
е
0oldpi/l1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*"
_class
loc:@oldpi/l1/kernel*
valueB"      *
dtype0
Ч
.oldpi/l1/kernel/Initializer/random_uniform/minConst*"
_class
loc:@oldpi/l1/kernel*
valueB
 *░ю╛*
dtype0*
_output_shapes
: 
Ч
.oldpi/l1/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@oldpi/l1/kernel*
valueB
 *░ю>*
dtype0*
_output_shapes
: 
я
8oldpi/l1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l1/kernel/Initializer/random_uniform/shape*

seed *
T0*"
_class
loc:@oldpi/l1/kernel*
seed2 *
dtype0*
_output_shapes
:	А
┌
.oldpi/l1/kernel/Initializer/random_uniform/subSub.oldpi/l1/kernel/Initializer/random_uniform/max.oldpi/l1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l1/kernel*
_output_shapes
: 
э
.oldpi/l1/kernel/Initializer/random_uniform/mulMul8oldpi/l1/kernel/Initializer/random_uniform/RandomUniform.oldpi/l1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@oldpi/l1/kernel*
_output_shapes
:	А
▀
*oldpi/l1/kernel/Initializer/random_uniformAdd.oldpi/l1/kernel/Initializer/random_uniform/mul.oldpi/l1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l1/kernel*
_output_shapes
:	А
й
oldpi/l1/kernel
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *"
_class
loc:@oldpi/l1/kernel*
	container *
shape:	А
╘
oldpi/l1/kernel/AssignAssignoldpi/l1/kernel*oldpi/l1/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*"
_class
loc:@oldpi/l1/kernel

oldpi/l1/kernel/readIdentityoldpi/l1/kernel*"
_class
loc:@oldpi/l1/kernel*
_output_shapes
:	А*
T0
Р
oldpi/l1/bias/Initializer/zerosConst* 
_class
loc:@oldpi/l1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Э
oldpi/l1/bias
VariableV2* 
_class
loc:@oldpi/l1/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
┐
oldpi/l1/bias/AssignAssignoldpi/l1/biasoldpi/l1/bias/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0* 
_class
loc:@oldpi/l1/bias*
validate_shape(
u
oldpi/l1/bias/readIdentityoldpi/l1/bias*
T0* 
_class
loc:@oldpi/l1/bias*
_output_shapes	
:А
П
oldpi/l1/MatMulMatMulstateoldpi/l1/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
К
oldpi/l1/BiasAddBiasAddoldpi/l1/MatMuloldpi/l1/bias/read*
data_formatNHWC*(
_output_shapes
:         А*
T0
Z
oldpi/l1/ReluReluoldpi/l1/BiasAdd*
T0*(
_output_shapes
:         А
е
0oldpi/l2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@oldpi/l2/kernel*
valueB"      
Ч
.oldpi/l2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *"
_class
loc:@oldpi/l2/kernel*
valueB
 *╫│▌╜*
dtype0
Ч
.oldpi/l2/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@oldpi/l2/kernel*
valueB
 *╫│▌=*
dtype0*
_output_shapes
: 
Ё
8oldpi/l2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l2/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
АА*

seed *
T0*"
_class
loc:@oldpi/l2/kernel*
seed2 
┌
.oldpi/l2/kernel/Initializer/random_uniform/subSub.oldpi/l2/kernel/Initializer/random_uniform/max.oldpi/l2/kernel/Initializer/random_uniform/min*"
_class
loc:@oldpi/l2/kernel*
_output_shapes
: *
T0
ю
.oldpi/l2/kernel/Initializer/random_uniform/mulMul8oldpi/l2/kernel/Initializer/random_uniform/RandomUniform.oldpi/l2/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АА*
T0*"
_class
loc:@oldpi/l2/kernel
р
*oldpi/l2/kernel/Initializer/random_uniformAdd.oldpi/l2/kernel/Initializer/random_uniform/mul.oldpi/l2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l2/kernel* 
_output_shapes
:
АА
л
oldpi/l2/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *"
_class
loc:@oldpi/l2/kernel*
	container *
shape:
АА
╒
oldpi/l2/kernel/AssignAssignoldpi/l2/kernel*oldpi/l2/kernel/Initializer/random_uniform*
T0*"
_class
loc:@oldpi/l2/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
А
oldpi/l2/kernel/readIdentityoldpi/l2/kernel*
T0*"
_class
loc:@oldpi/l2/kernel* 
_output_shapes
:
АА
Р
oldpi/l2/bias/Initializer/zerosConst* 
_class
loc:@oldpi/l2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Э
oldpi/l2/bias
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name * 
_class
loc:@oldpi/l2/bias*
	container 
┐
oldpi/l2/bias/AssignAssignoldpi/l2/biasoldpi/l2/bias/Initializer/zeros* 
_class
loc:@oldpi/l2/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
u
oldpi/l2/bias/readIdentityoldpi/l2/bias*
T0* 
_class
loc:@oldpi/l2/bias*
_output_shapes	
:А
Ч
oldpi/l2/MatMulMatMuloldpi/l1/Reluoldpi/l2/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
К
oldpi/l2/BiasAddBiasAddoldpi/l2/MatMuloldpi/l2/bias/read*(
_output_shapes
:         А*
T0*
data_formatNHWC
Z
oldpi/l2/ReluReluoldpi/l2/BiasAdd*
T0*(
_output_shapes
:         А
е
0oldpi/l3/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@oldpi/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ч
.oldpi/l3/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *"
_class
loc:@oldpi/l3/kernel*
valueB
 *╫│▌╜*
dtype0
Ч
.oldpi/l3/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@oldpi/l3/kernel*
valueB
 *╫│▌=*
dtype0*
_output_shapes
: 
Ё
8oldpi/l3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l3/kernel/Initializer/random_uniform/shape*

seed *
T0*"
_class
loc:@oldpi/l3/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА
┌
.oldpi/l3/kernel/Initializer/random_uniform/subSub.oldpi/l3/kernel/Initializer/random_uniform/max.oldpi/l3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@oldpi/l3/kernel
ю
.oldpi/l3/kernel/Initializer/random_uniform/mulMul8oldpi/l3/kernel/Initializer/random_uniform/RandomUniform.oldpi/l3/kernel/Initializer/random_uniform/sub*"
_class
loc:@oldpi/l3/kernel* 
_output_shapes
:
АА*
T0
р
*oldpi/l3/kernel/Initializer/random_uniformAdd.oldpi/l3/kernel/Initializer/random_uniform/mul.oldpi/l3/kernel/Initializer/random_uniform/min*"
_class
loc:@oldpi/l3/kernel* 
_output_shapes
:
АА*
T0
л
oldpi/l3/kernel
VariableV2*
shared_name *"
_class
loc:@oldpi/l3/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
╒
oldpi/l3/kernel/AssignAssignoldpi/l3/kernel*oldpi/l3/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*"
_class
loc:@oldpi/l3/kernel
А
oldpi/l3/kernel/readIdentityoldpi/l3/kernel*
T0*"
_class
loc:@oldpi/l3/kernel* 
_output_shapes
:
АА
Р
oldpi/l3/bias/Initializer/zerosConst* 
_class
loc:@oldpi/l3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Э
oldpi/l3/bias
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name * 
_class
loc:@oldpi/l3/bias*
	container 
┐
oldpi/l3/bias/AssignAssignoldpi/l3/biasoldpi/l3/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@oldpi/l3/bias*
validate_shape(*
_output_shapes	
:А
u
oldpi/l3/bias/readIdentityoldpi/l3/bias*
T0* 
_class
loc:@oldpi/l3/bias*
_output_shapes	
:А
Ч
oldpi/l3/MatMulMatMuloldpi/l2/Reluoldpi/l3/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
К
oldpi/l3/BiasAddBiasAddoldpi/l3/MatMuloldpi/l3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
Z
oldpi/l3/ReluReluoldpi/l3/BiasAdd*
T0*(
_output_shapes
:         А
е
0oldpi/l4/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@oldpi/l4/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
Ч
.oldpi/l4/kernel/Initializer/random_uniform/minConst*"
_class
loc:@oldpi/l4/kernel*
valueB
 *   ╛*
dtype0*
_output_shapes
: 
Ч
.oldpi/l4/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@oldpi/l4/kernel*
valueB
 *   >*
dtype0
Ё
8oldpi/l4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l4/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed *
T0*"
_class
loc:@oldpi/l4/kernel
┌
.oldpi/l4/kernel/Initializer/random_uniform/subSub.oldpi/l4/kernel/Initializer/random_uniform/max.oldpi/l4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l4/kernel*
_output_shapes
: 
ю
.oldpi/l4/kernel/Initializer/random_uniform/mulMul8oldpi/l4/kernel/Initializer/random_uniform/RandomUniform.oldpi/l4/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@oldpi/l4/kernel* 
_output_shapes
:
АА
р
*oldpi/l4/kernel/Initializer/random_uniformAdd.oldpi/l4/kernel/Initializer/random_uniform/mul.oldpi/l4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l4/kernel* 
_output_shapes
:
АА
л
oldpi/l4/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *"
_class
loc:@oldpi/l4/kernel*
	container *
shape:
АА
╒
oldpi/l4/kernel/AssignAssignoldpi/l4/kernel*oldpi/l4/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@oldpi/l4/kernel*
validate_shape(* 
_output_shapes
:
АА
А
oldpi/l4/kernel/readIdentityoldpi/l4/kernel*
T0*"
_class
loc:@oldpi/l4/kernel* 
_output_shapes
:
АА
Р
oldpi/l4/bias/Initializer/zerosConst*
_output_shapes	
:А* 
_class
loc:@oldpi/l4/bias*
valueBА*    *
dtype0
Э
oldpi/l4/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name * 
_class
loc:@oldpi/l4/bias*
	container *
shape:А
┐
oldpi/l4/bias/AssignAssignoldpi/l4/biasoldpi/l4/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@oldpi/l4/bias*
validate_shape(*
_output_shapes	
:А
u
oldpi/l4/bias/readIdentityoldpi/l4/bias*
_output_shapes	
:А*
T0* 
_class
loc:@oldpi/l4/bias
Ч
oldpi/l4/MatMulMatMuloldpi/l3/Reluoldpi/l4/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         А*
transpose_a( 
К
oldpi/l4/BiasAddBiasAddoldpi/l4/MatMuloldpi/l4/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
Z
oldpi/l4/ReluReluoldpi/l4/BiasAdd*(
_output_shapes
:         А*
T0
г
/oldpi/a/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@oldpi/a/kernel*
valueB"А      *
dtype0*
_output_shapes
:
Х
-oldpi/a/kernel/Initializer/random_uniform/minConst*!
_class
loc:@oldpi/a/kernel*
valueB
 *JQZ╛*
dtype0*
_output_shapes
: 
Х
-oldpi/a/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@oldpi/a/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
ь
7oldpi/a/kernel/Initializer/random_uniform/RandomUniformRandomUniform/oldpi/a/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А*

seed *
T0*!
_class
loc:@oldpi/a/kernel*
seed2 
╓
-oldpi/a/kernel/Initializer/random_uniform/subSub-oldpi/a/kernel/Initializer/random_uniform/max-oldpi/a/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@oldpi/a/kernel
щ
-oldpi/a/kernel/Initializer/random_uniform/mulMul7oldpi/a/kernel/Initializer/random_uniform/RandomUniform-oldpi/a/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@oldpi/a/kernel*
_output_shapes
:	А
█
)oldpi/a/kernel/Initializer/random_uniformAdd-oldpi/a/kernel/Initializer/random_uniform/mul-oldpi/a/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@oldpi/a/kernel*
_output_shapes
:	А
з
oldpi/a/kernel
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *!
_class
loc:@oldpi/a/kernel
╨
oldpi/a/kernel/AssignAssignoldpi/a/kernel)oldpi/a/kernel/Initializer/random_uniform*
_output_shapes
:	А*
use_locking(*
T0*!
_class
loc:@oldpi/a/kernel*
validate_shape(
|
oldpi/a/kernel/readIdentityoldpi/a/kernel*
T0*!
_class
loc:@oldpi/a/kernel*
_output_shapes
:	А
М
oldpi/a/bias/Initializer/zerosConst*
_class
loc:@oldpi/a/bias*
valueB*    *
dtype0*
_output_shapes
:
Щ
oldpi/a/bias
VariableV2*
shared_name *
_class
loc:@oldpi/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
║
oldpi/a/bias/AssignAssignoldpi/a/biasoldpi/a/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@oldpi/a/bias*
validate_shape(*
_output_shapes
:
q
oldpi/a/bias/readIdentityoldpi/a/bias*
T0*
_class
loc:@oldpi/a/bias*
_output_shapes
:
Ф
oldpi/a/MatMulMatMuloldpi/l4/Reluoldpi/a/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
Ж
oldpi/a/BiasAddBiasAddoldpi/a/MatMuloldpi/a/bias/read*'
_output_shapes
:         *
T0*
data_formatNHWC
W
oldpi/a/TanhTanholdpi/a/BiasAdd*
T0*'
_output_shapes
:         
л
3oldpi/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
_class
loc:@oldpi/dense/kernel*
valueB"А      *
dtype0
Э
1oldpi/dense/kernel/Initializer/random_uniform/minConst*%
_class
loc:@oldpi/dense/kernel*
valueB
 *JQZ╛*
dtype0*
_output_shapes
: 
Э
1oldpi/dense/kernel/Initializer/random_uniform/maxConst*%
_class
loc:@oldpi/dense/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
°
;oldpi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform3oldpi/dense/kernel/Initializer/random_uniform/shape*
T0*%
_class
loc:@oldpi/dense/kernel*
seed2 *
dtype0*
_output_shapes
:	А*

seed 
ц
1oldpi/dense/kernel/Initializer/random_uniform/subSub1oldpi/dense/kernel/Initializer/random_uniform/max1oldpi/dense/kernel/Initializer/random_uniform/min*
T0*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
: 
∙
1oldpi/dense/kernel/Initializer/random_uniform/mulMul;oldpi/dense/kernel/Initializer/random_uniform/RandomUniform1oldpi/dense/kernel/Initializer/random_uniform/sub*
T0*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
:	А
ы
-oldpi/dense/kernel/Initializer/random_uniformAdd1oldpi/dense/kernel/Initializer/random_uniform/mul1oldpi/dense/kernel/Initializer/random_uniform/min*
T0*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
:	А
п
oldpi/dense/kernel
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *%
_class
loc:@oldpi/dense/kernel
р
oldpi/dense/kernel/AssignAssignoldpi/dense/kernel-oldpi/dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*%
_class
loc:@oldpi/dense/kernel
И
oldpi/dense/kernel/readIdentityoldpi/dense/kernel*
T0*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
:	А
Ф
"oldpi/dense/bias/Initializer/zerosConst*
_output_shapes
:*#
_class
loc:@oldpi/dense/bias*
valueB*    *
dtype0
б
oldpi/dense/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *#
_class
loc:@oldpi/dense/bias*
	container 
╩
oldpi/dense/bias/AssignAssignoldpi/dense/bias"oldpi/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*#
_class
loc:@oldpi/dense/bias
}
oldpi/dense/bias/readIdentityoldpi/dense/bias*
T0*#
_class
loc:@oldpi/dense/bias*
_output_shapes
:
Ь
oldpi/dense/MatMulMatMuloldpi/l4/Reluoldpi/dense/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
Т
oldpi/dense/BiasAddBiasAddoldpi/dense/MatMuloldpi/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
g
oldpi/dense/SoftplusSoftplusoldpi/dense/BiasAdd*
T0*'
_output_shapes
:         
\
oldpi/Normal/locIdentityoldpi/a/Tanh*
T0*'
_output_shapes
:         
f
oldpi/Normal/scaleIdentityoldpi/dense/Softplus*
T0*'
_output_shapes
:         
_
pi/Normal/sample/sample_shapeConst*
value	B :*
dtype0*
_output_shapes
: 
i
pi/Normal/sample/sample_shape_1Const*
valueB:*
dtype0*
_output_shapes
:
o
"pi/Normal/batch_shape_tensor/ShapeShapepi/Normal/loc*
T0*
out_type0*
_output_shapes
:
s
$pi/Normal/batch_shape_tensor/Shape_1Shapepi/Normal/scale*
T0*
out_type0*
_output_shapes
:
к
*pi/Normal/batch_shape_tensor/BroadcastArgsBroadcastArgs"pi/Normal/batch_shape_tensor/Shape$pi/Normal/batch_shape_tensor/Shape_1*
T0*
_output_shapes
:
j
 pi/Normal/sample/concat/values_0Const*
valueB:*
dtype0*
_output_shapes
:
^
pi/Normal/sample/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╔
pi/Normal/sample/concatConcatV2 pi/Normal/sample/concat/values_0*pi/Normal/batch_shape_tensor/BroadcastArgspi/Normal/sample/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
h
#pi/Normal/sample/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
j
%pi/Normal/sample/random_normal/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╔
3pi/Normal/sample/random_normal/RandomStandardNormalRandomStandardNormalpi/Normal/sample/concat*
dtype0*4
_output_shapes"
 :                  *
seed2 *

seed *
T0
─
"pi/Normal/sample/random_normal/mulMul3pi/Normal/sample/random_normal/RandomStandardNormal%pi/Normal/sample/random_normal/stddev*
T0*4
_output_shapes"
 :                  
н
pi/Normal/sample/random_normalAdd"pi/Normal/sample/random_normal/mul#pi/Normal/sample/random_normal/mean*
T0*4
_output_shapes"
 :                  
В
pi/Normal/sample/mulMulpi/Normal/sample/random_normalpi/Normal/scale*+
_output_shapes
:         *
T0
v
pi/Normal/sample/addAddpi/Normal/sample/mulpi/Normal/loc*
T0*+
_output_shapes
:         
j
pi/Normal/sample/ShapeShapepi/Normal/sample/add*
out_type0*
_output_shapes
:*
T0
n
$pi/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
p
&pi/Normal/sample/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
p
&pi/Normal/sample/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╥
pi/Normal/sample/strided_sliceStridedSlicepi/Normal/sample/Shape$pi/Normal/sample/strided_slice/stack&pi/Normal/sample/strided_slice/stack_1&pi/Normal/sample/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0
`
pi/Normal/sample/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
└
pi/Normal/sample/concat_1ConcatV2pi/Normal/sample/sample_shape_1pi/Normal/sample/strided_slicepi/Normal/sample/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
Ш
pi/Normal/sample/ReshapeReshapepi/Normal/sample/addpi/Normal/sample/concat_1*+
_output_shapes
:         *
T0*
Tshape0
Г
sample_action/SqueezeSqueezepi/Normal/sample/Reshape*
squeeze_dims
 *
T0*'
_output_shapes
:         
╕
update_oldpi/AssignAssignoldpi/l1/kernelpi/l1/kernel/read*
_output_shapes
:	А*
use_locking( *
T0*"
_class
loc:@oldpi/l1/kernel*
validate_shape(
░
update_oldpi/Assign_1Assignoldpi/l1/biaspi/l1/bias/read* 
_class
loc:@oldpi/l1/bias*
validate_shape(*
_output_shapes	
:А*
use_locking( *
T0
╗
update_oldpi/Assign_2Assignoldpi/l2/kernelpi/l2/kernel/read*"
_class
loc:@oldpi/l2/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking( *
T0
░
update_oldpi/Assign_3Assignoldpi/l2/biaspi/l2/bias/read*
use_locking( *
T0* 
_class
loc:@oldpi/l2/bias*
validate_shape(*
_output_shapes	
:А
╗
update_oldpi/Assign_4Assignoldpi/l3/kernelpi/l3/kernel/read*"
_class
loc:@oldpi/l3/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking( *
T0
░
update_oldpi/Assign_5Assignoldpi/l3/biaspi/l3/bias/read* 
_class
loc:@oldpi/l3/bias*
validate_shape(*
_output_shapes	
:А*
use_locking( *
T0
╗
update_oldpi/Assign_6Assignoldpi/l4/kernelpi/l4/kernel/read*
use_locking( *
T0*"
_class
loc:@oldpi/l4/kernel*
validate_shape(* 
_output_shapes
:
АА
░
update_oldpi/Assign_7Assignoldpi/l4/biaspi/l4/bias/read*
use_locking( *
T0* 
_class
loc:@oldpi/l4/bias*
validate_shape(*
_output_shapes	
:А
╖
update_oldpi/Assign_8Assignoldpi/a/kernelpi/a/kernel/read*
_output_shapes
:	А*
use_locking( *
T0*!
_class
loc:@oldpi/a/kernel*
validate_shape(
м
update_oldpi/Assign_9Assignoldpi/a/biaspi/a/bias/read*
_output_shapes
:*
use_locking( *
T0*
_class
loc:@oldpi/a/bias*
validate_shape(
─
update_oldpi/Assign_10Assignoldpi/dense/kernelpi/dense/kernel/read*
use_locking( *
T0*%
_class
loc:@oldpi/dense/kernel*
validate_shape(*
_output_shapes
:	А
╣
update_oldpi/Assign_11Assignoldpi/dense/biaspi/dense/bias/read*#
_class
loc:@oldpi/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking( *
T0
i
actionPlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
l
	advantagePlaceholder*
shape:         *
dtype0*'
_output_shapes
:         
n
pi/Normal/prob/standardize/subSubactionpi/Normal/loc*
T0*'
_output_shapes
:         
Р
"pi/Normal/prob/standardize/truedivRealDivpi/Normal/prob/standardize/subpi/Normal/scale*
T0*'
_output_shapes
:         
u
pi/Normal/prob/SquareSquare"pi/Normal/prob/standardize/truediv*
T0*'
_output_shapes
:         
Y
pi/Normal/prob/mul/xConst*
valueB
 *   ┐*
dtype0*
_output_shapes
: 
x
pi/Normal/prob/mulMulpi/Normal/prob/mul/xpi/Normal/prob/Square*
T0*'
_output_shapes
:         
\
pi/Normal/prob/LogLogpi/Normal/scale*'
_output_shapes
:         *
T0
Y
pi/Normal/prob/add/xConst*
_output_shapes
: *
valueB
 *О?k?*
dtype0
u
pi/Normal/prob/addAddpi/Normal/prob/add/xpi/Normal/prob/Log*'
_output_shapes
:         *
T0
s
pi/Normal/prob/subSubpi/Normal/prob/mulpi/Normal/prob/add*
T0*'
_output_shapes
:         
_
pi/Normal/prob/ExpExppi/Normal/prob/sub*'
_output_shapes
:         *
T0
t
!oldpi/Normal/prob/standardize/subSubactionoldpi/Normal/loc*
T0*'
_output_shapes
:         
Щ
%oldpi/Normal/prob/standardize/truedivRealDiv!oldpi/Normal/prob/standardize/suboldpi/Normal/scale*'
_output_shapes
:         *
T0
{
oldpi/Normal/prob/SquareSquare%oldpi/Normal/prob/standardize/truediv*'
_output_shapes
:         *
T0
\
oldpi/Normal/prob/mul/xConst*
_output_shapes
: *
valueB
 *   ┐*
dtype0
Б
oldpi/Normal/prob/mulMuloldpi/Normal/prob/mul/xoldpi/Normal/prob/Square*'
_output_shapes
:         *
T0
b
oldpi/Normal/prob/LogLogoldpi/Normal/scale*'
_output_shapes
:         *
T0
\
oldpi/Normal/prob/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *О?k?
~
oldpi/Normal/prob/addAddoldpi/Normal/prob/add/xoldpi/Normal/prob/Log*
T0*'
_output_shapes
:         
|
oldpi/Normal/prob/subSuboldpi/Normal/prob/muloldpi/Normal/prob/add*'
_output_shapes
:         *
T0
e
oldpi/Normal/prob/ExpExpoldpi/Normal/prob/sub*
T0*'
_output_shapes
:         
~
loss/surrogate/truedivRealDivpi/Normal/prob/Expoldpi/Normal/prob/Exp*
T0*'
_output_shapes
:         
n
loss/surrogate/mulMulloss/surrogate/truediv	advantage*
T0*'
_output_shapes
:         
a
loss/clip_by_value/Minimum/yConst*
valueB
 *ЪЩЩ?*
dtype0*
_output_shapes
: 
Н
loss/clip_by_value/MinimumMinimumloss/surrogate/truedivloss/clip_by_value/Minimum/y*
T0*'
_output_shapes
:         
Y
loss/clip_by_value/yConst*
valueB
 *═╠L?*
dtype0*
_output_shapes
: 
Б
loss/clip_by_valueMaximumloss/clip_by_value/Minimumloss/clip_by_value/y*
T0*'
_output_shapes
:         
`
loss/mulMulloss/clip_by_value	advantage*'
_output_shapes
:         *
T0
g
loss/MinimumMinimumloss/surrogate/mulloss/mul*'
_output_shapes
:         *
T0
[

loss/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
i
	loss/MeanMeanloss/Minimum
loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
;
loss/NegNeg	loss/Mean*
_output_shapes
: *
T0
Y
atrain/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
_
atrain/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Д
atrain/gradients/FillFillatrain/gradients/Shapeatrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
a
"atrain/gradients/loss/Neg_grad/NegNegatrain/gradients/Fill*
_output_shapes
: *
T0
~
-atrain/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
╝
'atrain/gradients/loss/Mean_grad/ReshapeReshape"atrain/gradients/loss/Neg_grad/Neg-atrain/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
q
%atrain/gradients/loss/Mean_grad/ShapeShapeloss/Minimum*
T0*
out_type0*
_output_shapes
:
└
$atrain/gradients/loss/Mean_grad/TileTile'atrain/gradients/loss/Mean_grad/Reshape%atrain/gradients/loss/Mean_grad/Shape*'
_output_shapes
:         *

Tmultiples0*
T0
s
'atrain/gradients/loss/Mean_grad/Shape_1Shapeloss/Minimum*
T0*
out_type0*
_output_shapes
:
j
'atrain/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
o
%atrain/gradients/loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
║
$atrain/gradients/loss/Mean_grad/ProdProd'atrain/gradients/loss/Mean_grad/Shape_1%atrain/gradients/loss/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
q
'atrain/gradients/loss/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
╛
&atrain/gradients/loss/Mean_grad/Prod_1Prod'atrain/gradients/loss/Mean_grad/Shape_2'atrain/gradients/loss/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
k
)atrain/gradients/loss/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
ж
'atrain/gradients/loss/Mean_grad/MaximumMaximum&atrain/gradients/loss/Mean_grad/Prod_1)atrain/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
д
(atrain/gradients/loss/Mean_grad/floordivFloorDiv$atrain/gradients/loss/Mean_grad/Prod'atrain/gradients/loss/Mean_grad/Maximum*
_output_shapes
: *
T0
Ж
$atrain/gradients/loss/Mean_grad/CastCast(atrain/gradients/loss/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
░
'atrain/gradients/loss/Mean_grad/truedivRealDiv$atrain/gradients/loss/Mean_grad/Tile$atrain/gradients/loss/Mean_grad/Cast*'
_output_shapes
:         *
T0
z
(atrain/gradients/loss/Minimum_grad/ShapeShapeloss/surrogate/mul*
_output_shapes
:*
T0*
out_type0
r
*atrain/gradients/loss/Minimum_grad/Shape_1Shapeloss/mul*
T0*
out_type0*
_output_shapes
:
С
*atrain/gradients/loss/Minimum_grad/Shape_2Shape'atrain/gradients/loss/Mean_grad/truediv*
T0*
out_type0*
_output_shapes
:
s
.atrain/gradients/loss/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╨
(atrain/gradients/loss/Minimum_grad/zerosFill*atrain/gradients/loss/Minimum_grad/Shape_2.atrain/gradients/loss/Minimum_grad/zeros/Const*

index_type0*'
_output_shapes
:         *
T0
Й
,atrain/gradients/loss/Minimum_grad/LessEqual	LessEqualloss/surrogate/mulloss/mul*
T0*'
_output_shapes
:         
ф
8atrain/gradients/loss/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs(atrain/gradients/loss/Minimum_grad/Shape*atrain/gradients/loss/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ц
)atrain/gradients/loss/Minimum_grad/SelectSelect,atrain/gradients/loss/Minimum_grad/LessEqual'atrain/gradients/loss/Mean_grad/truediv(atrain/gradients/loss/Minimum_grad/zeros*'
_output_shapes
:         *
T0
ш
+atrain/gradients/loss/Minimum_grad/Select_1Select,atrain/gradients/loss/Minimum_grad/LessEqual(atrain/gradients/loss/Minimum_grad/zeros'atrain/gradients/loss/Mean_grad/truediv*'
_output_shapes
:         *
T0
╥
&atrain/gradients/loss/Minimum_grad/SumSum)atrain/gradients/loss/Minimum_grad/Select8atrain/gradients/loss/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╟
*atrain/gradients/loss/Minimum_grad/ReshapeReshape&atrain/gradients/loss/Minimum_grad/Sum(atrain/gradients/loss/Minimum_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
╪
(atrain/gradients/loss/Minimum_grad/Sum_1Sum+atrain/gradients/loss/Minimum_grad/Select_1:atrain/gradients/loss/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
═
,atrain/gradients/loss/Minimum_grad/Reshape_1Reshape(atrain/gradients/loss/Minimum_grad/Sum_1*atrain/gradients/loss/Minimum_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
Ч
3atrain/gradients/loss/Minimum_grad/tuple/group_depsNoOp+^atrain/gradients/loss/Minimum_grad/Reshape-^atrain/gradients/loss/Minimum_grad/Reshape_1
Ъ
;atrain/gradients/loss/Minimum_grad/tuple/control_dependencyIdentity*atrain/gradients/loss/Minimum_grad/Reshape4^atrain/gradients/loss/Minimum_grad/tuple/group_deps*
T0*=
_class3
1/loc:@atrain/gradients/loss/Minimum_grad/Reshape*'
_output_shapes
:         
а
=atrain/gradients/loss/Minimum_grad/tuple/control_dependency_1Identity,atrain/gradients/loss/Minimum_grad/Reshape_14^atrain/gradients/loss/Minimum_grad/tuple/group_deps*'
_output_shapes
:         *
T0*?
_class5
31loc:@atrain/gradients/loss/Minimum_grad/Reshape_1
Д
.atrain/gradients/loss/surrogate/mul_grad/ShapeShapeloss/surrogate/truediv*
out_type0*
_output_shapes
:*
T0
y
0atrain/gradients/loss/surrogate/mul_grad/Shape_1Shape	advantage*
T0*
out_type0*
_output_shapes
:
Ў
>atrain/gradients/loss/surrogate/mul_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/loss/surrogate/mul_grad/Shape0atrain/gradients/loss/surrogate/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
н
,atrain/gradients/loss/surrogate/mul_grad/MulMul;atrain/gradients/loss/Minimum_grad/tuple/control_dependency	advantage*'
_output_shapes
:         *
T0
с
,atrain/gradients/loss/surrogate/mul_grad/SumSum,atrain/gradients/loss/surrogate/mul_grad/Mul>atrain/gradients/loss/surrogate/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
┘
0atrain/gradients/loss/surrogate/mul_grad/ReshapeReshape,atrain/gradients/loss/surrogate/mul_grad/Sum.atrain/gradients/loss/surrogate/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
╝
.atrain/gradients/loss/surrogate/mul_grad/Mul_1Mulloss/surrogate/truediv;atrain/gradients/loss/Minimum_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
ч
.atrain/gradients/loss/surrogate/mul_grad/Sum_1Sum.atrain/gradients/loss/surrogate/mul_grad/Mul_1@atrain/gradients/loss/surrogate/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
▀
2atrain/gradients/loss/surrogate/mul_grad/Reshape_1Reshape.atrain/gradients/loss/surrogate/mul_grad/Sum_10atrain/gradients/loss/surrogate/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
й
9atrain/gradients/loss/surrogate/mul_grad/tuple/group_depsNoOp1^atrain/gradients/loss/surrogate/mul_grad/Reshape3^atrain/gradients/loss/surrogate/mul_grad/Reshape_1
▓
Aatrain/gradients/loss/surrogate/mul_grad/tuple/control_dependencyIdentity0atrain/gradients/loss/surrogate/mul_grad/Reshape:^atrain/gradients/loss/surrogate/mul_grad/tuple/group_deps*'
_output_shapes
:         *
T0*C
_class9
75loc:@atrain/gradients/loss/surrogate/mul_grad/Reshape
╕
Catrain/gradients/loss/surrogate/mul_grad/tuple/control_dependency_1Identity2atrain/gradients/loss/surrogate/mul_grad/Reshape_1:^atrain/gradients/loss/surrogate/mul_grad/tuple/group_deps*'
_output_shapes
:         *
T0*E
_class;
97loc:@atrain/gradients/loss/surrogate/mul_grad/Reshape_1
v
$atrain/gradients/loss/mul_grad/ShapeShapeloss/clip_by_value*
T0*
out_type0*
_output_shapes
:
o
&atrain/gradients/loss/mul_grad/Shape_1Shape	advantage*
_output_shapes
:*
T0*
out_type0
╪
4atrain/gradients/loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs$atrain/gradients/loss/mul_grad/Shape&atrain/gradients/loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
е
"atrain/gradients/loss/mul_grad/MulMul=atrain/gradients/loss/Minimum_grad/tuple/control_dependency_1	advantage*
T0*'
_output_shapes
:         
├
"atrain/gradients/loss/mul_grad/SumSum"atrain/gradients/loss/mul_grad/Mul4atrain/gradients/loss/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╗
&atrain/gradients/loss/mul_grad/ReshapeReshape"atrain/gradients/loss/mul_grad/Sum$atrain/gradients/loss/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
░
$atrain/gradients/loss/mul_grad/Mul_1Mulloss/clip_by_value=atrain/gradients/loss/Minimum_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
╔
$atrain/gradients/loss/mul_grad/Sum_1Sum$atrain/gradients/loss/mul_grad/Mul_16atrain/gradients/loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┴
(atrain/gradients/loss/mul_grad/Reshape_1Reshape$atrain/gradients/loss/mul_grad/Sum_1&atrain/gradients/loss/mul_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
Л
/atrain/gradients/loss/mul_grad/tuple/group_depsNoOp'^atrain/gradients/loss/mul_grad/Reshape)^atrain/gradients/loss/mul_grad/Reshape_1
К
7atrain/gradients/loss/mul_grad/tuple/control_dependencyIdentity&atrain/gradients/loss/mul_grad/Reshape0^atrain/gradients/loss/mul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@atrain/gradients/loss/mul_grad/Reshape*'
_output_shapes
:         
Р
9atrain/gradients/loss/mul_grad/tuple/control_dependency_1Identity(atrain/gradients/loss/mul_grad/Reshape_10^atrain/gradients/loss/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@atrain/gradients/loss/mul_grad/Reshape_1*'
_output_shapes
:         
И
.atrain/gradients/loss/clip_by_value_grad/ShapeShapeloss/clip_by_value/Minimum*
_output_shapes
:*
T0*
out_type0
s
0atrain/gradients/loss/clip_by_value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
з
0atrain/gradients/loss/clip_by_value_grad/Shape_2Shape7atrain/gradients/loss/mul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
y
4atrain/gradients/loss/clip_by_value_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
т
.atrain/gradients/loss/clip_by_value_grad/zerosFill0atrain/gradients/loss/clip_by_value_grad/Shape_24atrain/gradients/loss/clip_by_value_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:         
й
5atrain/gradients/loss/clip_by_value_grad/GreaterEqualGreaterEqualloss/clip_by_value/Minimumloss/clip_by_value/y*'
_output_shapes
:         *
T0
Ў
>atrain/gradients/loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/loss/clip_by_value_grad/Shape0atrain/gradients/loss/clip_by_value_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Л
/atrain/gradients/loss/clip_by_value_grad/SelectSelect5atrain/gradients/loss/clip_by_value_grad/GreaterEqual7atrain/gradients/loss/mul_grad/tuple/control_dependency.atrain/gradients/loss/clip_by_value_grad/zeros*'
_output_shapes
:         *
T0
Н
1atrain/gradients/loss/clip_by_value_grad/Select_1Select5atrain/gradients/loss/clip_by_value_grad/GreaterEqual.atrain/gradients/loss/clip_by_value_grad/zeros7atrain/gradients/loss/mul_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
ф
,atrain/gradients/loss/clip_by_value_grad/SumSum/atrain/gradients/loss/clip_by_value_grad/Select>atrain/gradients/loss/clip_by_value_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
┘
0atrain/gradients/loss/clip_by_value_grad/ReshapeReshape,atrain/gradients/loss/clip_by_value_grad/Sum.atrain/gradients/loss/clip_by_value_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
ъ
.atrain/gradients/loss/clip_by_value_grad/Sum_1Sum1atrain/gradients/loss/clip_by_value_grad/Select_1@atrain/gradients/loss/clip_by_value_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╬
2atrain/gradients/loss/clip_by_value_grad/Reshape_1Reshape.atrain/gradients/loss/clip_by_value_grad/Sum_10atrain/gradients/loss/clip_by_value_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
й
9atrain/gradients/loss/clip_by_value_grad/tuple/group_depsNoOp1^atrain/gradients/loss/clip_by_value_grad/Reshape3^atrain/gradients/loss/clip_by_value_grad/Reshape_1
▓
Aatrain/gradients/loss/clip_by_value_grad/tuple/control_dependencyIdentity0atrain/gradients/loss/clip_by_value_grad/Reshape:^atrain/gradients/loss/clip_by_value_grad/tuple/group_deps*'
_output_shapes
:         *
T0*C
_class9
75loc:@atrain/gradients/loss/clip_by_value_grad/Reshape
з
Catrain/gradients/loss/clip_by_value_grad/tuple/control_dependency_1Identity2atrain/gradients/loss/clip_by_value_grad/Reshape_1:^atrain/gradients/loss/clip_by_value_grad/tuple/group_deps*
_output_shapes
: *
T0*E
_class;
97loc:@atrain/gradients/loss/clip_by_value_grad/Reshape_1
М
6atrain/gradients/loss/clip_by_value/Minimum_grad/ShapeShapeloss/surrogate/truediv*
T0*
out_type0*
_output_shapes
:
{
8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
╣
8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_2ShapeAatrain/gradients/loss/clip_by_value_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Б
<atrain/gradients/loss/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
·
6atrain/gradients/loss/clip_by_value/Minimum_grad/zerosFill8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_2<atrain/gradients/loss/clip_by_value/Minimum_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:         
п
:atrain/gradients/loss/clip_by_value/Minimum_grad/LessEqual	LessEqualloss/surrogate/truedivloss/clip_by_value/Minimum/y*'
_output_shapes
:         *
T0
О
Fatrain/gradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs6atrain/gradients/loss/clip_by_value/Minimum_grad/Shape8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:         :         
к
7atrain/gradients/loss/clip_by_value/Minimum_grad/SelectSelect:atrain/gradients/loss/clip_by_value/Minimum_grad/LessEqualAatrain/gradients/loss/clip_by_value_grad/tuple/control_dependency6atrain/gradients/loss/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:         *
T0
м
9atrain/gradients/loss/clip_by_value/Minimum_grad/Select_1Select:atrain/gradients/loss/clip_by_value/Minimum_grad/LessEqual6atrain/gradients/loss/clip_by_value/Minimum_grad/zerosAatrain/gradients/loss/clip_by_value_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
№
4atrain/gradients/loss/clip_by_value/Minimum_grad/SumSum7atrain/gradients/loss/clip_by_value/Minimum_grad/SelectFatrain/gradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ё
8atrain/gradients/loss/clip_by_value/Minimum_grad/ReshapeReshape4atrain/gradients/loss/clip_by_value/Minimum_grad/Sum6atrain/gradients/loss/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
В
6atrain/gradients/loss/clip_by_value/Minimum_grad/Sum_1Sum9atrain/gradients/loss/clip_by_value/Minimum_grad/Select_1Hatrain/gradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ц
:atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1Reshape6atrain/gradients/loss/clip_by_value/Minimum_grad/Sum_18atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
┴
Aatrain/gradients/loss/clip_by_value/Minimum_grad/tuple/group_depsNoOp9^atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape;^atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1
╥
Iatrain/gradients/loss/clip_by_value/Minimum_grad/tuple/control_dependencyIdentity8atrain/gradients/loss/clip_by_value/Minimum_grad/ReshapeB^atrain/gradients/loss/clip_by_value/Minimum_grad/tuple/group_deps*
T0*K
_classA
?=loc:@atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape*'
_output_shapes
:         
╟
Katrain/gradients/loss/clip_by_value/Minimum_grad/tuple/control_dependency_1Identity:atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1B^atrain/gradients/loss/clip_by_value/Minimum_grad/tuple/group_deps*
T0*M
_classC
A?loc:@atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1*
_output_shapes
: 
л
atrain/gradients/AddNAddNAatrain/gradients/loss/surrogate/mul_grad/tuple/control_dependencyIatrain/gradients/loss/clip_by_value/Minimum_grad/tuple/control_dependency*
T0*C
_class9
75loc:@atrain/gradients/loss/surrogate/mul_grad/Reshape*
N*'
_output_shapes
:         
Д
2atrain/gradients/loss/surrogate/truediv_grad/ShapeShapepi/Normal/prob/Exp*
T0*
out_type0*
_output_shapes
:
Й
4atrain/gradients/loss/surrogate/truediv_grad/Shape_1Shapeoldpi/Normal/prob/Exp*
_output_shapes
:*
T0*
out_type0
В
Batrain/gradients/loss/surrogate/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs2atrain/gradients/loss/surrogate/truediv_grad/Shape4atrain/gradients/loss/surrogate/truediv_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Я
4atrain/gradients/loss/surrogate/truediv_grad/RealDivRealDivatrain/gradients/AddNoldpi/Normal/prob/Exp*
T0*'
_output_shapes
:         
ё
0atrain/gradients/loss/surrogate/truediv_grad/SumSum4atrain/gradients/loss/surrogate/truediv_grad/RealDivBatrain/gradients/loss/surrogate/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
х
4atrain/gradients/loss/surrogate/truediv_grad/ReshapeReshape0atrain/gradients/loss/surrogate/truediv_grad/Sum2atrain/gradients/loss/surrogate/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
}
0atrain/gradients/loss/surrogate/truediv_grad/NegNegpi/Normal/prob/Exp*
T0*'
_output_shapes
:         
╝
6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_1RealDiv0atrain/gradients/loss/surrogate/truediv_grad/Negoldpi/Normal/prob/Exp*
T0*'
_output_shapes
:         
┬
6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_2RealDiv6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_1oldpi/Normal/prob/Exp*
T0*'
_output_shapes
:         
╕
0atrain/gradients/loss/surrogate/truediv_grad/mulMulatrain/gradients/AddN6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:         
ё
2atrain/gradients/loss/surrogate/truediv_grad/Sum_1Sum0atrain/gradients/loss/surrogate/truediv_grad/mulDatrain/gradients/loss/surrogate/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ы
6atrain/gradients/loss/surrogate/truediv_grad/Reshape_1Reshape2atrain/gradients/loss/surrogate/truediv_grad/Sum_14atrain/gradients/loss/surrogate/truediv_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
╡
=atrain/gradients/loss/surrogate/truediv_grad/tuple/group_depsNoOp5^atrain/gradients/loss/surrogate/truediv_grad/Reshape7^atrain/gradients/loss/surrogate/truediv_grad/Reshape_1
┬
Eatrain/gradients/loss/surrogate/truediv_grad/tuple/control_dependencyIdentity4atrain/gradients/loss/surrogate/truediv_grad/Reshape>^atrain/gradients/loss/surrogate/truediv_grad/tuple/group_deps*'
_output_shapes
:         *
T0*G
_class=
;9loc:@atrain/gradients/loss/surrogate/truediv_grad/Reshape
╚
Gatrain/gradients/loss/surrogate/truediv_grad/tuple/control_dependency_1Identity6atrain/gradients/loss/surrogate/truediv_grad/Reshape_1>^atrain/gradients/loss/surrogate/truediv_grad/tuple/group_deps*
T0*I
_class?
=;loc:@atrain/gradients/loss/surrogate/truediv_grad/Reshape_1*'
_output_shapes
:         
└
,atrain/gradients/pi/Normal/prob/Exp_grad/mulMulEatrain/gradients/loss/surrogate/truediv_grad/tuple/control_dependencypi/Normal/prob/Exp*
T0*'
_output_shapes
:         
А
.atrain/gradients/pi/Normal/prob/sub_grad/ShapeShapepi/Normal/prob/mul*
T0*
out_type0*
_output_shapes
:
В
0atrain/gradients/pi/Normal/prob/sub_grad/Shape_1Shapepi/Normal/prob/add*
T0*
out_type0*
_output_shapes
:
Ў
>atrain/gradients/pi/Normal/prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/pi/Normal/prob/sub_grad/Shape0atrain/gradients/pi/Normal/prob/sub_grad/Shape_1*2
_output_shapes 
:         :         *
T0
с
,atrain/gradients/pi/Normal/prob/sub_grad/SumSum,atrain/gradients/pi/Normal/prob/Exp_grad/mul>atrain/gradients/pi/Normal/prob/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┘
0atrain/gradients/pi/Normal/prob/sub_grad/ReshapeReshape,atrain/gradients/pi/Normal/prob/sub_grad/Sum.atrain/gradients/pi/Normal/prob/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
х
.atrain/gradients/pi/Normal/prob/sub_grad/Sum_1Sum,atrain/gradients/pi/Normal/prob/Exp_grad/mul@atrain/gradients/pi/Normal/prob/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ж
,atrain/gradients/pi/Normal/prob/sub_grad/NegNeg.atrain/gradients/pi/Normal/prob/sub_grad/Sum_1*
T0*
_output_shapes
:
▌
2atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1Reshape,atrain/gradients/pi/Normal/prob/sub_grad/Neg0atrain/gradients/pi/Normal/prob/sub_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
й
9atrain/gradients/pi/Normal/prob/sub_grad/tuple/group_depsNoOp1^atrain/gradients/pi/Normal/prob/sub_grad/Reshape3^atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1
▓
Aatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependencyIdentity0atrain/gradients/pi/Normal/prob/sub_grad/Reshape:^atrain/gradients/pi/Normal/prob/sub_grad/tuple/group_deps*
T0*C
_class9
75loc:@atrain/gradients/pi/Normal/prob/sub_grad/Reshape*'
_output_shapes
:         
╕
Catrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1:^atrain/gradients/pi/Normal/prob/sub_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1*'
_output_shapes
:         
q
.atrain/gradients/pi/Normal/prob/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Е
0atrain/gradients/pi/Normal/prob/mul_grad/Shape_1Shapepi/Normal/prob/Square*
out_type0*
_output_shapes
:*
T0
Ў
>atrain/gradients/pi/Normal/prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/pi/Normal/prob/mul_grad/Shape0atrain/gradients/pi/Normal/prob/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
┐
,atrain/gradients/pi/Normal/prob/mul_grad/MulMulAatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependencypi/Normal/prob/Square*
T0*'
_output_shapes
:         
с
,atrain/gradients/pi/Normal/prob/mul_grad/SumSum,atrain/gradients/pi/Normal/prob/mul_grad/Mul>atrain/gradients/pi/Normal/prob/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╚
0atrain/gradients/pi/Normal/prob/mul_grad/ReshapeReshape,atrain/gradients/pi/Normal/prob/mul_grad/Sum.atrain/gradients/pi/Normal/prob/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
└
.atrain/gradients/pi/Normal/prob/mul_grad/Mul_1Mulpi/Normal/prob/mul/xAatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
ч
.atrain/gradients/pi/Normal/prob/mul_grad/Sum_1Sum.atrain/gradients/pi/Normal/prob/mul_grad/Mul_1@atrain/gradients/pi/Normal/prob/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
▀
2atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1Reshape.atrain/gradients/pi/Normal/prob/mul_grad/Sum_10atrain/gradients/pi/Normal/prob/mul_grad/Shape_1*
Tshape0*'
_output_shapes
:         *
T0
й
9atrain/gradients/pi/Normal/prob/mul_grad/tuple/group_depsNoOp1^atrain/gradients/pi/Normal/prob/mul_grad/Reshape3^atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1
б
Aatrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependencyIdentity0atrain/gradients/pi/Normal/prob/mul_grad/Reshape:^atrain/gradients/pi/Normal/prob/mul_grad/tuple/group_deps*
T0*C
_class9
75loc:@atrain/gradients/pi/Normal/prob/mul_grad/Reshape*
_output_shapes
: 
╕
Catrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1:^atrain/gradients/pi/Normal/prob/mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1*'
_output_shapes
:         
q
.atrain/gradients/pi/Normal/prob/add_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
В
0atrain/gradients/pi/Normal/prob/add_grad/Shape_1Shapepi/Normal/prob/Log*
T0*
out_type0*
_output_shapes
:
Ў
>atrain/gradients/pi/Normal/prob/add_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/pi/Normal/prob/add_grad/Shape0atrain/gradients/pi/Normal/prob/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
°
,atrain/gradients/pi/Normal/prob/add_grad/SumSumCatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency_1>atrain/gradients/pi/Normal/prob/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╚
0atrain/gradients/pi/Normal/prob/add_grad/ReshapeReshape,atrain/gradients/pi/Normal/prob/add_grad/Sum.atrain/gradients/pi/Normal/prob/add_grad/Shape*
_output_shapes
: *
T0*
Tshape0
№
.atrain/gradients/pi/Normal/prob/add_grad/Sum_1SumCatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency_1@atrain/gradients/pi/Normal/prob/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
▀
2atrain/gradients/pi/Normal/prob/add_grad/Reshape_1Reshape.atrain/gradients/pi/Normal/prob/add_grad/Sum_10atrain/gradients/pi/Normal/prob/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
й
9atrain/gradients/pi/Normal/prob/add_grad/tuple/group_depsNoOp1^atrain/gradients/pi/Normal/prob/add_grad/Reshape3^atrain/gradients/pi/Normal/prob/add_grad/Reshape_1
б
Aatrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependencyIdentity0atrain/gradients/pi/Normal/prob/add_grad/Reshape:^atrain/gradients/pi/Normal/prob/add_grad/tuple/group_deps*
T0*C
_class9
75loc:@atrain/gradients/pi/Normal/prob/add_grad/Reshape*
_output_shapes
: 
╕
Catrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/Normal/prob/add_grad/Reshape_1:^atrain/gradients/pi/Normal/prob/add_grad/tuple/group_deps*E
_class;
97loc:@atrain/gradients/pi/Normal/prob/add_grad/Reshape_1*'
_output_shapes
:         *
T0
╝
1atrain/gradients/pi/Normal/prob/Square_grad/ConstConstD^atrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
┐
/atrain/gradients/pi/Normal/prob/Square_grad/MulMul"pi/Normal/prob/standardize/truediv1atrain/gradients/pi/Normal/prob/Square_grad/Const*'
_output_shapes
:         *
T0
р
1atrain/gradients/pi/Normal/prob/Square_grad/Mul_1MulCatrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependency_1/atrain/gradients/pi/Normal/prob/Square_grad/Mul*
T0*'
_output_shapes
:         
╩
3atrain/gradients/pi/Normal/prob/Log_grad/Reciprocal
Reciprocalpi/Normal/scaleD^atrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
▀
,atrain/gradients/pi/Normal/prob/Log_grad/mulMulCatrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependency_13atrain/gradients/pi/Normal/prob/Log_grad/Reciprocal*'
_output_shapes
:         *
T0
Ь
>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ShapeShapepi/Normal/prob/standardize/sub*
out_type0*
_output_shapes
:*
T0
П
@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape_1Shapepi/Normal/scale*
out_type0*
_output_shapes
:*
T0
ж
Natrain/gradients/pi/Normal/prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┴
@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDivRealDiv1atrain/gradients/pi/Normal/prob/Square_grad/Mul_1pi/Normal/scale*
T0*'
_output_shapes
:         
Х
<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/SumSum@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDivNatrain/gradients/pi/Normal/prob/standardize/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Й
@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ReshapeReshape<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Sum>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Х
<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/NegNegpi/Normal/prob/standardize/sub*'
_output_shapes
:         *
T0
╬
Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_1RealDiv<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Negpi/Normal/scale*
T0*'
_output_shapes
:         
╘
Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_2RealDivBatrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_1pi/Normal/scale*
T0*'
_output_shapes
:         
ь
<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/mulMul1atrain/gradients/pi/Normal/prob/Square_grad/Mul_1Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:         
Х
>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Sum_1Sum<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/mulPatrain/gradients/pi/Normal/prob/standardize/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
П
Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1Reshape>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Sum_1@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
┘
Iatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/group_depsNoOpA^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ReshapeC^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1
Є
Qatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependencyIdentity@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ReshapeJ^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/group_deps*'
_output_shapes
:         *
T0*S
_classI
GEloc:@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape
°
Satrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependency_1IdentityBatrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1J^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/group_deps*
T0*U
_classK
IGloc:@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1*'
_output_shapes
:         
А
:atrain/gradients/pi/Normal/prob/standardize/sub_grad/ShapeShapeaction*
_output_shapes
:*
T0*
out_type0
Й
<atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape_1Shapepi/Normal/loc*
T0*
out_type0*
_output_shapes
:
Ъ
Jatrain/gradients/pi/Normal/prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape<atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ю
8atrain/gradients/pi/Normal/prob/standardize/sub_grad/SumSumQatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependencyJatrain/gradients/pi/Normal/prob/standardize/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
¤
<atrain/gradients/pi/Normal/prob/standardize/sub_grad/ReshapeReshape8atrain/gradients/pi/Normal/prob/standardize/sub_grad/Sum:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
в
:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Sum_1SumQatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependencyLatrain/gradients/pi/Normal/prob/standardize/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ю
8atrain/gradients/pi/Normal/prob/standardize/sub_grad/NegNeg:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:
Б
>atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1Reshape8atrain/gradients/pi/Normal/prob/standardize/sub_grad/Neg<atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
═
Eatrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/group_depsNoOp=^atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape?^atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1
т
Matrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/control_dependencyIdentity<atrain/gradients/pi/Normal/prob/standardize/sub_grad/ReshapeF^atrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/group_deps*
T0*O
_classE
CAloc:@atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape*'
_output_shapes
:         
ш
Oatrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/control_dependency_1Identity>atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1F^atrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1*'
_output_shapes
:         
Ю
atrain/gradients/AddN_1AddN,atrain/gradients/pi/Normal/prob/Log_grad/mulSatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependency_1*
T0*?
_class5
31loc:@atrain/gradients/pi/Normal/prob/Log_grad/mul*
N*'
_output_shapes
:         
б
4atrain/gradients/pi/dense/Softplus_grad/SoftplusGradSoftplusGradatrain/gradients/AddN_1pi/dense/BiasAdd*
T0*'
_output_shapes
:         
┬
(atrain/gradients/pi/a/Tanh_grad/TanhGradTanhGrad	pi/a/TanhOatrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
│
2atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad4atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad*
T0*
data_formatNHWC*
_output_shapes
:
л
7atrain/gradients/pi/dense/BiasAdd_grad/tuple/group_depsNoOp3^atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad
╢
?atrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity4atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad8^atrain/gradients/pi/dense/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:         *
T0*G
_class=
;9loc:@atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad
з
Aatrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGrad8^atrain/gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*E
_class;
97loc:@atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGrad
г
.atrain/gradients/pi/a/BiasAdd_grad/BiasAddGradBiasAddGrad(atrain/gradients/pi/a/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:
Ч
3atrain/gradients/pi/a/BiasAdd_grad/tuple/group_depsNoOp/^atrain/gradients/pi/a/BiasAdd_grad/BiasAddGrad)^atrain/gradients/pi/a/Tanh_grad/TanhGrad
Ц
;atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependencyIdentity(atrain/gradients/pi/a/Tanh_grad/TanhGrad4^atrain/gradients/pi/a/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:         *
T0*;
_class1
/-loc:@atrain/gradients/pi/a/Tanh_grad/TanhGrad
Ч
=atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependency_1Identity.atrain/gradients/pi/a/BiasAdd_grad/BiasAddGrad4^atrain/gradients/pi/a/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@atrain/gradients/pi/a/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ц
,atrain/gradients/pi/dense/MatMul_grad/MatMulMatMul?atrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependencypi/dense/kernel/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b(
╒
.atrain/gradients/pi/dense/MatMul_grad/MatMul_1MatMul
pi/l4/Relu?atrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	А*
transpose_a(*
transpose_b( *
T0
Ю
6atrain/gradients/pi/dense/MatMul_grad/tuple/group_depsNoOp-^atrain/gradients/pi/dense/MatMul_grad/MatMul/^atrain/gradients/pi/dense/MatMul_grad/MatMul_1
е
>atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependencyIdentity,atrain/gradients/pi/dense/MatMul_grad/MatMul7^atrain/gradients/pi/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@atrain/gradients/pi/dense/MatMul_grad/MatMul*(
_output_shapes
:         А
в
@atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Identity.atrain/gradients/pi/dense/MatMul_grad/MatMul_17^atrain/gradients/pi/dense/MatMul_grad/tuple/group_deps*
_output_shapes
:	А*
T0*A
_class7
53loc:@atrain/gradients/pi/dense/MatMul_grad/MatMul_1
┌
(atrain/gradients/pi/a/MatMul_grad/MatMulMatMul;atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependencypi/a/kernel/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b(
═
*atrain/gradients/pi/a/MatMul_grad/MatMul_1MatMul
pi/l4/Relu;atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	А*
transpose_a(*
transpose_b( 
Т
2atrain/gradients/pi/a/MatMul_grad/tuple/group_depsNoOp)^atrain/gradients/pi/a/MatMul_grad/MatMul+^atrain/gradients/pi/a/MatMul_grad/MatMul_1
Х
:atrain/gradients/pi/a/MatMul_grad/tuple/control_dependencyIdentity(atrain/gradients/pi/a/MatMul_grad/MatMul3^atrain/gradients/pi/a/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@atrain/gradients/pi/a/MatMul_grad/MatMul*(
_output_shapes
:         А*
T0
Т
<atrain/gradients/pi/a/MatMul_grad/tuple/control_dependency_1Identity*atrain/gradients/pi/a/MatMul_grad/MatMul_13^atrain/gradients/pi/a/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@atrain/gradients/pi/a/MatMul_grad/MatMul_1*
_output_shapes
:	А
Ш
atrain/gradients/AddN_2AddN>atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependency:atrain/gradients/pi/a/MatMul_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0*?
_class5
31loc:@atrain/gradients/pi/dense/MatMul_grad/MatMul*
N
Н
)atrain/gradients/pi/l4/Relu_grad/ReluGradReluGradatrain/gradients/AddN_2
pi/l4/Relu*(
_output_shapes
:         А*
T0
ж
/atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l4/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:А*
T0
Ъ
4atrain/gradients/pi/l4/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l4/Relu_grad/ReluGrad
Ы
<atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l4/Relu_grad/ReluGrad5^atrain/gradients/pi/l4/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l4/Relu_grad/ReluGrad*(
_output_shapes
:         А
Ь
>atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l4/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*B
_class8
64loc:@atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGrad
▌
)atrain/gradients/pi/l4/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependencypi/l4/kernel/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b(
╨
+atrain/gradients/pi/l4/MatMul_grad/MatMul_1MatMul
pi/l3/Relu<atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( *
T0
Х
3atrain/gradients/pi/l4/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l4/MatMul_grad/MatMul,^atrain/gradients/pi/l4/MatMul_grad/MatMul_1
Щ
;atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l4/MatMul_grad/MatMul4^atrain/gradients/pi/l4/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l4/MatMul_grad/MatMul*(
_output_shapes
:         А
Ч
=atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l4/MatMul_grad/MatMul_14^atrain/gradients/pi/l4/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@atrain/gradients/pi/l4/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
▒
)atrain/gradients/pi/l3/Relu_grad/ReluGradReluGrad;atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependency
pi/l3/Relu*
T0*(
_output_shapes
:         А
ж
/atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Ъ
4atrain/gradients/pi/l3/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l3/Relu_grad/ReluGrad
Ы
<atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l3/Relu_grad/ReluGrad5^atrain/gradients/pi/l3/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l3/Relu_grad/ReluGrad*(
_output_shapes
:         А
Ь
>atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l3/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А*
T0
▌
)atrain/gradients/pi/l3/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependencypi/l3/kernel/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b(
╨
+atrain/gradients/pi/l3/MatMul_grad/MatMul_1MatMul
pi/l2/Relu<atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( 
Х
3atrain/gradients/pi/l3/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l3/MatMul_grad/MatMul,^atrain/gradients/pi/l3/MatMul_grad/MatMul_1
Щ
;atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l3/MatMul_grad/MatMul4^atrain/gradients/pi/l3/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l3/MatMul_grad/MatMul*(
_output_shapes
:         А
Ч
=atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l3/MatMul_grad/MatMul_14^atrain/gradients/pi/l3/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@atrain/gradients/pi/l3/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
▒
)atrain/gradients/pi/l2/Relu_grad/ReluGradReluGrad;atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependency
pi/l2/Relu*(
_output_shapes
:         А*
T0
ж
/atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Ъ
4atrain/gradients/pi/l2/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l2/Relu_grad/ReluGrad
Ы
<atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l2/Relu_grad/ReluGrad5^atrain/gradients/pi/l2/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@atrain/gradients/pi/l2/Relu_grad/ReluGrad*(
_output_shapes
:         А*
T0
Ь
>atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l2/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
▌
)atrain/gradients/pi/l2/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependencypi/l2/kernel/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b(
╨
+atrain/gradients/pi/l2/MatMul_grad/MatMul_1MatMul
pi/l1/Relu<atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( 
Х
3atrain/gradients/pi/l2/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l2/MatMul_grad/MatMul,^atrain/gradients/pi/l2/MatMul_grad/MatMul_1
Щ
;atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l2/MatMul_grad/MatMul4^atrain/gradients/pi/l2/MatMul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*<
_class2
0.loc:@atrain/gradients/pi/l2/MatMul_grad/MatMul
Ч
=atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l2/MatMul_grad/MatMul_14^atrain/gradients/pi/l2/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@atrain/gradients/pi/l2/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
▒
)atrain/gradients/pi/l1/Relu_grad/ReluGradReluGrad;atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependency
pi/l1/Relu*
T0*(
_output_shapes
:         А
ж
/atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:А*
T0
Ъ
4atrain/gradients/pi/l1/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l1/Relu_grad/ReluGrad
Ы
<atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l1/Relu_grad/ReluGrad5^atrain/gradients/pi/l1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l1/Relu_grad/ReluGrad*(
_output_shapes
:         А
Ь
>atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*B
_class8
64loc:@atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGrad
▄
)atrain/gradients/pi/l1/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependencypi/l1/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b(
╩
+atrain/gradients/pi/l1/MatMul_grad/MatMul_1MatMulstate<atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	А*
transpose_a(*
transpose_b( *
T0
Х
3atrain/gradients/pi/l1/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l1/MatMul_grad/MatMul,^atrain/gradients/pi/l1/MatMul_grad/MatMul_1
Ш
;atrain/gradients/pi/l1/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l1/MatMul_grad/MatMul4^atrain/gradients/pi/l1/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l1/MatMul_grad/MatMul*'
_output_shapes
:         
Ц
=atrain/gradients/pi/l1/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l1/MatMul_grad/MatMul_14^atrain/gradients/pi/l1/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@atrain/gradients/pi/l1/MatMul_grad/MatMul_1*
_output_shapes
:	А
Г
 atrain/beta1_power/initial_valueConst*
_output_shapes
: *
_class
loc:@pi/a/bias*
valueB
 *fff?*
dtype0
Ф
atrain/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@pi/a/bias*
	container *
shape: 
┴
atrain/beta1_power/AssignAssignatrain/beta1_power atrain/beta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@pi/a/bias
v
atrain/beta1_power/readIdentityatrain/beta1_power*
_class
loc:@pi/a/bias*
_output_shapes
: *
T0
Г
 atrain/beta2_power/initial_valueConst*
_class
loc:@pi/a/bias*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
Ф
atrain/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@pi/a/bias*
	container *
shape: 
┴
atrain/beta2_power/AssignAssignatrain/beta2_power atrain/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@pi/a/bias
v
atrain/beta2_power/readIdentityatrain/beta2_power*
T0*
_class
loc:@pi/a/bias*
_output_shapes
: 
м
:atrain/pi/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ц
0atrain/pi/l1/kernel/Adam/Initializer/zeros/ConstConst*
_class
loc:@pi/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
*atrain/pi/l1/kernel/Adam/Initializer/zerosFill:atrain/pi/l1/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l1/kernel/Adam/Initializer/zeros/Const*
_class
loc:@pi/l1/kernel*

index_type0*
_output_shapes
:	А*
T0
п
atrain/pi/l1/kernel/Adam
VariableV2*
shared_name *
_class
loc:@pi/l1/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
у
atrain/pi/l1/kernel/Adam/AssignAssignatrain/pi/l1/kernel/Adam*atrain/pi/l1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l1/kernel*
validate_shape(*
_output_shapes
:	А
О
atrain/pi/l1/kernel/Adam/readIdentityatrain/pi/l1/kernel/Adam*
_class
loc:@pi/l1/kernel*
_output_shapes
:	А*
T0
о
<atrain/pi/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@pi/l1/kernel*
valueB"      
Ш
2atrain/pi/l1/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@pi/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Г
,atrain/pi/l1/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l1/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	А*
T0*
_class
loc:@pi/l1/kernel*

index_type0
▒
atrain/pi/l1/kernel/Adam_1
VariableV2*
_class
loc:@pi/l1/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name 
щ
!atrain/pi/l1/kernel/Adam_1/AssignAssignatrain/pi/l1/kernel/Adam_1,atrain/pi/l1/kernel/Adam_1/Initializer/zeros*
_class
loc:@pi/l1/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0
Т
atrain/pi/l1/kernel/Adam_1/readIdentityatrain/pi/l1/kernel/Adam_1*
T0*
_class
loc:@pi/l1/kernel*
_output_shapes
:	А
Ц
(atrain/pi/l1/bias/Adam/Initializer/zerosConst*
_class
loc:@pi/l1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
г
atrain/pi/l1/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l1/bias*
	container *
shape:А
╫
atrain/pi/l1/bias/Adam/AssignAssignatrain/pi/l1/bias/Adam(atrain/pi/l1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@pi/l1/bias
Д
atrain/pi/l1/bias/Adam/readIdentityatrain/pi/l1/bias/Adam*
_class
loc:@pi/l1/bias*
_output_shapes	
:А*
T0
Ш
*atrain/pi/l1/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
е
atrain/pi/l1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l1/bias*
	container *
shape:А
▌
atrain/pi/l1/bias/Adam_1/AssignAssignatrain/pi/l1/bias/Adam_1*atrain/pi/l1/bias/Adam_1/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@pi/l1/bias*
validate_shape(
И
atrain/pi/l1/bias/Adam_1/readIdentityatrain/pi/l1/bias/Adam_1*
T0*
_class
loc:@pi/l1/bias*
_output_shapes	
:А
м
:atrain/pi/l2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@pi/l2/kernel*
valueB"      
Ц
0atrain/pi/l2/kernel/Adam/Initializer/zeros/ConstConst*
_class
loc:@pi/l2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
■
*atrain/pi/l2/kernel/Adam/Initializer/zerosFill:atrain/pi/l2/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l2/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
АА*
T0*
_class
loc:@pi/l2/kernel*

index_type0
▒
atrain/pi/l2/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *
_class
loc:@pi/l2/kernel*
	container *
shape:
АА
ф
atrain/pi/l2/kernel/Adam/AssignAssignatrain/pi/l2/kernel/Adam*atrain/pi/l2/kernel/Adam/Initializer/zeros*
T0*
_class
loc:@pi/l2/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
П
atrain/pi/l2/kernel/Adam/readIdentityatrain/pi/l2/kernel/Adam*
T0*
_class
loc:@pi/l2/kernel* 
_output_shapes
:
АА
о
<atrain/pi/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
_class
loc:@pi/l2/kernel*
valueB"      *
dtype0
Ш
2atrain/pi/l2/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
_class
loc:@pi/l2/kernel*
valueB
 *    *
dtype0
Д
,atrain/pi/l2/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l2/kernel/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@pi/l2/kernel*

index_type0* 
_output_shapes
:
АА
│
atrain/pi/l2/kernel/Adam_1
VariableV2*
shared_name *
_class
loc:@pi/l2/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
ъ
!atrain/pi/l2/kernel/Adam_1/AssignAssignatrain/pi/l2/kernel/Adam_1,atrain/pi/l2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l2/kernel*
validate_shape(* 
_output_shapes
:
АА
У
atrain/pi/l2/kernel/Adam_1/readIdentityatrain/pi/l2/kernel/Adam_1* 
_output_shapes
:
АА*
T0*
_class
loc:@pi/l2/kernel
Ц
(atrain/pi/l2/bias/Adam/Initializer/zerosConst*
_class
loc:@pi/l2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
г
atrain/pi/l2/bias/Adam
VariableV2*
shared_name *
_class
loc:@pi/l2/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
╫
atrain/pi/l2/bias/Adam/AssignAssignatrain/pi/l2/bias/Adam(atrain/pi/l2/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l2/bias*
validate_shape(*
_output_shapes	
:А
Д
atrain/pi/l2/bias/Adam/readIdentityatrain/pi/l2/bias/Adam*
_output_shapes	
:А*
T0*
_class
loc:@pi/l2/bias
Ш
*atrain/pi/l2/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
е
atrain/pi/l2/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@pi/l2/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
▌
atrain/pi/l2/bias/Adam_1/AssignAssignatrain/pi/l2/bias/Adam_1*atrain/pi/l2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l2/bias*
validate_shape(*
_output_shapes	
:А
И
atrain/pi/l2/bias/Adam_1/readIdentityatrain/pi/l2/bias/Adam_1*
T0*
_class
loc:@pi/l2/bias*
_output_shapes	
:А
м
:atrain/pi/l3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ц
0atrain/pi/l3/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@pi/l3/kernel*
valueB
 *    
■
*atrain/pi/l3/kernel/Adam/Initializer/zerosFill:atrain/pi/l3/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l3/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
АА*
T0*
_class
loc:@pi/l3/kernel*

index_type0
▒
atrain/pi/l3/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *
_class
loc:@pi/l3/kernel*
	container *
shape:
АА
ф
atrain/pi/l3/kernel/Adam/AssignAssignatrain/pi/l3/kernel/Adam*atrain/pi/l3/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l3/kernel*
validate_shape(* 
_output_shapes
:
АА
П
atrain/pi/l3/kernel/Adam/readIdentityatrain/pi/l3/kernel/Adam*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:
АА
о
<atrain/pi/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ш
2atrain/pi/l3/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@pi/l3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Д
,atrain/pi/l3/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l3/kernel/Adam_1/Initializer/zeros/Const*
_class
loc:@pi/l3/kernel*

index_type0* 
_output_shapes
:
АА*
T0
│
atrain/pi/l3/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *
_class
loc:@pi/l3/kernel*
	container *
shape:
АА
ъ
!atrain/pi/l3/kernel/Adam_1/AssignAssignatrain/pi/l3/kernel/Adam_1,atrain/pi/l3/kernel/Adam_1/Initializer/zeros*
T0*
_class
loc:@pi/l3/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
У
atrain/pi/l3/kernel/Adam_1/readIdentityatrain/pi/l3/kernel/Adam_1*
_class
loc:@pi/l3/kernel* 
_output_shapes
:
АА*
T0
Ц
(atrain/pi/l3/bias/Adam/Initializer/zerosConst*
_output_shapes	
:А*
_class
loc:@pi/l3/bias*
valueBА*    *
dtype0
г
atrain/pi/l3/bias/Adam
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l3/bias*
	container 
╫
atrain/pi/l3/bias/Adam/AssignAssignatrain/pi/l3/bias/Adam(atrain/pi/l3/bias/Adam/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@pi/l3/bias*
validate_shape(
Д
atrain/pi/l3/bias/Adam/readIdentityatrain/pi/l3/bias/Adam*
_class
loc:@pi/l3/bias*
_output_shapes	
:А*
T0
Ш
*atrain/pi/l3/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
е
atrain/pi/l3/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l3/bias*
	container *
shape:А
▌
atrain/pi/l3/bias/Adam_1/AssignAssignatrain/pi/l3/bias/Adam_1*atrain/pi/l3/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l3/bias*
validate_shape(*
_output_shapes	
:А
И
atrain/pi/l3/bias/Adam_1/readIdentityatrain/pi/l3/bias/Adam_1*
T0*
_class
loc:@pi/l3/bias*
_output_shapes	
:А
м
:atrain/pi/l4/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l4/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
Ц
0atrain/pi/l4/kernel/Adam/Initializer/zeros/ConstConst*
_class
loc:@pi/l4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
■
*atrain/pi/l4/kernel/Adam/Initializer/zerosFill:atrain/pi/l4/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l4/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
АА*
T0*
_class
loc:@pi/l4/kernel*

index_type0
▒
atrain/pi/l4/kernel/Adam
VariableV2*
_class
loc:@pi/l4/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name 
ф
atrain/pi/l4/kernel/Adam/AssignAssignatrain/pi/l4/kernel/Adam*atrain/pi/l4/kernel/Adam/Initializer/zeros* 
_output_shapes
:
АА*
use_locking(*
T0*
_class
loc:@pi/l4/kernel*
validate_shape(
П
atrain/pi/l4/kernel/Adam/readIdentityatrain/pi/l4/kernel/Adam*
T0*
_class
loc:@pi/l4/kernel* 
_output_shapes
:
АА
о
<atrain/pi/l4/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l4/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
Ш
2atrain/pi/l4/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@pi/l4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Д
,atrain/pi/l4/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l4/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l4/kernel/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@pi/l4/kernel*

index_type0* 
_output_shapes
:
АА
│
atrain/pi/l4/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *
_class
loc:@pi/l4/kernel*
	container *
shape:
АА
ъ
!atrain/pi/l4/kernel/Adam_1/AssignAssignatrain/pi/l4/kernel/Adam_1,atrain/pi/l4/kernel/Adam_1/Initializer/zeros*
_class
loc:@pi/l4/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0
У
atrain/pi/l4/kernel/Adam_1/readIdentityatrain/pi/l4/kernel/Adam_1*
T0*
_class
loc:@pi/l4/kernel* 
_output_shapes
:
АА
Ц
(atrain/pi/l4/bias/Adam/Initializer/zerosConst*
_class
loc:@pi/l4/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
г
atrain/pi/l4/bias/Adam
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l4/bias
╫
atrain/pi/l4/bias/Adam/AssignAssignatrain/pi/l4/bias/Adam(atrain/pi/l4/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l4/bias*
validate_shape(*
_output_shapes	
:А
Д
atrain/pi/l4/bias/Adam/readIdentityatrain/pi/l4/bias/Adam*
T0*
_class
loc:@pi/l4/bias*
_output_shapes	
:А
Ш
*atrain/pi/l4/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l4/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
е
atrain/pi/l4/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@pi/l4/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
▌
atrain/pi/l4/bias/Adam_1/AssignAssignatrain/pi/l4/bias/Adam_1*atrain/pi/l4/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l4/bias*
validate_shape(*
_output_shapes	
:А
И
atrain/pi/l4/bias/Adam_1/readIdentityatrain/pi/l4/bias/Adam_1*
_output_shapes	
:А*
T0*
_class
loc:@pi/l4/bias
а
)atrain/pi/a/kernel/Adam/Initializer/zerosConst*
_class
loc:@pi/a/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
н
atrain/pi/a/kernel/Adam
VariableV2*
_class
loc:@pi/a/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name 
▀
atrain/pi/a/kernel/Adam/AssignAssignatrain/pi/a/kernel/Adam)atrain/pi/a/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/a/kernel*
validate_shape(*
_output_shapes
:	А
Л
atrain/pi/a/kernel/Adam/readIdentityatrain/pi/a/kernel/Adam*
T0*
_class
loc:@pi/a/kernel*
_output_shapes
:	А
в
+atrain/pi/a/kernel/Adam_1/Initializer/zerosConst*
_class
loc:@pi/a/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
п
atrain/pi/a/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *
_class
loc:@pi/a/kernel*
	container *
shape:	А
х
 atrain/pi/a/kernel/Adam_1/AssignAssignatrain/pi/a/kernel/Adam_1+atrain/pi/a/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/a/kernel*
validate_shape(*
_output_shapes
:	А
П
atrain/pi/a/kernel/Adam_1/readIdentityatrain/pi/a/kernel/Adam_1*
_output_shapes
:	А*
T0*
_class
loc:@pi/a/kernel
Т
'atrain/pi/a/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
_class
loc:@pi/a/bias*
valueB*    *
dtype0
Я
atrain/pi/a/bias/Adam
VariableV2*
shared_name *
_class
loc:@pi/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
╥
atrain/pi/a/bias/Adam/AssignAssignatrain/pi/a/bias/Adam'atrain/pi/a/bias/Adam/Initializer/zeros*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
А
atrain/pi/a/bias/Adam/readIdentityatrain/pi/a/bias/Adam*
T0*
_class
loc:@pi/a/bias*
_output_shapes
:
Ф
)atrain/pi/a/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@pi/a/bias*
valueB*    
б
atrain/pi/a/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@pi/a/bias*
	container *
shape:
╪
atrain/pi/a/bias/Adam_1/AssignAssignatrain/pi/a/bias/Adam_1)atrain/pi/a/bias/Adam_1/Initializer/zeros*
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Д
atrain/pi/a/bias/Adam_1/readIdentityatrain/pi/a/bias/Adam_1*
T0*
_class
loc:@pi/a/bias*
_output_shapes
:
и
-atrain/pi/dense/kernel/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
╡
atrain/pi/dense/kernel/Adam
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *"
_class
loc:@pi/dense/kernel
я
"atrain/pi/dense/kernel/Adam/AssignAssignatrain/pi/dense/kernel/Adam-atrain/pi/dense/kernel/Adam/Initializer/zeros*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(
Ч
 atrain/pi/dense/kernel/Adam/readIdentityatrain/pi/dense/kernel/Adam*
_output_shapes
:	А*
T0*"
_class
loc:@pi/dense/kernel
к
/atrain/pi/dense/kernel/Adam_1/Initializer/zerosConst*
_output_shapes
:	А*"
_class
loc:@pi/dense/kernel*
valueB	А*    *
dtype0
╖
atrain/pi/dense/kernel/Adam_1
VariableV2*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
ї
$atrain/pi/dense/kernel/Adam_1/AssignAssignatrain/pi/dense/kernel/Adam_1/atrain/pi/dense/kernel/Adam_1/Initializer/zeros*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(
Ы
"atrain/pi/dense/kernel/Adam_1/readIdentityatrain/pi/dense/kernel/Adam_1*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	А*
T0
Ъ
+atrain/pi/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
з
atrain/pi/dense/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@pi/dense/bias*
	container 
т
 atrain/pi/dense/bias/Adam/AssignAssignatrain/pi/dense/bias/Adam+atrain/pi/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:
М
atrain/pi/dense/bias/Adam/readIdentityatrain/pi/dense/bias/Adam*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:
Ь
-atrain/pi/dense/bias/Adam_1/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
й
atrain/pi/dense/bias/Adam_1
VariableV2* 
_class
loc:@pi/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
ш
"atrain/pi/dense/bias/Adam_1/AssignAssignatrain/pi/dense/bias/Adam_1-atrain/pi/dense/bias/Adam_1/Initializer/zeros*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Р
 atrain/pi/dense/bias/Adam_1/readIdentityatrain/pi/dense/bias/Adam_1*
_output_shapes
:*
T0* 
_class
loc:@pi/dense/bias
^
atrain/Adam/learning_rateConst*
valueB
 *╖╤8*
dtype0*
_output_shapes
: 
V
atrain/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
V
atrain/Adam/beta2Const*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
X
atrain/Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
│
)atrain/Adam/update_pi/l1/kernel/ApplyAdam	ApplyAdampi/l1/kernelatrain/pi/l1/kernel/Adamatrain/pi/l1/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l1/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	А*
use_locking( *
T0*
_class
loc:@pi/l1/kernel*
use_nesterov( 
ж
'atrain/Adam/update_pi/l1/bias/ApplyAdam	ApplyAdam
pi/l1/biasatrain/pi/l1/bias/Adamatrain/pi/l1/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l1/bias*
use_nesterov( *
_output_shapes	
:А
┤
)atrain/Adam/update_pi/l2/kernel/ApplyAdam	ApplyAdampi/l2/kernelatrain/pi/l2/kernel/Adamatrain/pi/l2/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l2/kernel*
use_nesterov( * 
_output_shapes
:
АА
ж
'atrain/Adam/update_pi/l2/bias/ApplyAdam	ApplyAdam
pi/l2/biasatrain/pi/l2/bias/Adamatrain/pi/l2/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@pi/l2/bias*
use_nesterov( *
_output_shapes	
:А*
use_locking( 
┤
)atrain/Adam/update_pi/l3/kernel/ApplyAdam	ApplyAdampi/l3/kernelatrain/pi/l3/kernel/Adamatrain/pi/l3/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l3/kernel*
use_nesterov( * 
_output_shapes
:
АА
ж
'atrain/Adam/update_pi/l3/bias/ApplyAdam	ApplyAdam
pi/l3/biasatrain/pi/l3/bias/Adamatrain/pi/l3/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l3/bias*
use_nesterov( *
_output_shapes	
:А
┤
)atrain/Adam/update_pi/l4/kernel/ApplyAdam	ApplyAdampi/l4/kernelatrain/pi/l4/kernel/Adamatrain/pi/l4/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l4/kernel*
use_nesterov( * 
_output_shapes
:
АА
ж
'atrain/Adam/update_pi/l4/bias/ApplyAdam	ApplyAdam
pi/l4/biasatrain/pi/l4/bias/Adamatrain/pi/l4/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l4/bias*
use_nesterov( *
_output_shapes	
:А
н
(atrain/Adam/update_pi/a/kernel/ApplyAdam	ApplyAdampi/a/kernelatrain/pi/a/kernel/Adamatrain/pi/a/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon<atrain/gradients/pi/a/MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@pi/a/kernel*
use_nesterov( *
_output_shapes
:	А*
use_locking( 
Я
&atrain/Adam/update_pi/a/bias/ApplyAdam	ApplyAdam	pi/a/biasatrain/pi/a/bias/Adamatrain/pi/a/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*
_class
loc:@pi/a/bias
┼
,atrain/Adam/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelatrain/pi/dense/kernel/Adamatrain/pi/dense/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon@atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@pi/dense/kernel*
use_nesterov( *
_output_shapes
:	А
╖
*atrain/Adam/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biasatrain/pi/dense/bias/Adamatrain/pi/dense/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilonAatrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@pi/dense/bias*
use_nesterov( *
_output_shapes
:
Й
atrain/Adam/mulMulatrain/beta1_power/readatrain/Adam/beta1'^atrain/Adam/update_pi/a/bias/ApplyAdam)^atrain/Adam/update_pi/a/kernel/ApplyAdam+^atrain/Adam/update_pi/dense/bias/ApplyAdam-^atrain/Adam/update_pi/dense/kernel/ApplyAdam(^atrain/Adam/update_pi/l1/bias/ApplyAdam*^atrain/Adam/update_pi/l1/kernel/ApplyAdam(^atrain/Adam/update_pi/l2/bias/ApplyAdam*^atrain/Adam/update_pi/l2/kernel/ApplyAdam(^atrain/Adam/update_pi/l3/bias/ApplyAdam*^atrain/Adam/update_pi/l3/kernel/ApplyAdam(^atrain/Adam/update_pi/l4/bias/ApplyAdam*^atrain/Adam/update_pi/l4/kernel/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@pi/a/bias
й
atrain/Adam/AssignAssignatrain/beta1_poweratrain/Adam/mul*
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
Л
atrain/Adam/mul_1Mulatrain/beta2_power/readatrain/Adam/beta2'^atrain/Adam/update_pi/a/bias/ApplyAdam)^atrain/Adam/update_pi/a/kernel/ApplyAdam+^atrain/Adam/update_pi/dense/bias/ApplyAdam-^atrain/Adam/update_pi/dense/kernel/ApplyAdam(^atrain/Adam/update_pi/l1/bias/ApplyAdam*^atrain/Adam/update_pi/l1/kernel/ApplyAdam(^atrain/Adam/update_pi/l2/bias/ApplyAdam*^atrain/Adam/update_pi/l2/kernel/ApplyAdam(^atrain/Adam/update_pi/l3/bias/ApplyAdam*^atrain/Adam/update_pi/l3/kernel/ApplyAdam(^atrain/Adam/update_pi/l4/bias/ApplyAdam*^atrain/Adam/update_pi/l4/kernel/ApplyAdam*
T0*
_class
loc:@pi/a/bias*
_output_shapes
: 
н
atrain/Adam/Assign_1Assignatrain/beta2_poweratrain/Adam/mul_1*
use_locking( *
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
: 
╟
atrain/AdamNoOp^atrain/Adam/Assign^atrain/Adam/Assign_1'^atrain/Adam/update_pi/a/bias/ApplyAdam)^atrain/Adam/update_pi/a/kernel/ApplyAdam+^atrain/Adam/update_pi/dense/bias/ApplyAdam-^atrain/Adam/update_pi/dense/kernel/ApplyAdam(^atrain/Adam/update_pi/l1/bias/ApplyAdam*^atrain/Adam/update_pi/l1/kernel/ApplyAdam(^atrain/Adam/update_pi/l2/bias/ApplyAdam*^atrain/Adam/update_pi/l2/kernel/ApplyAdam(^atrain/Adam/update_pi/l3/bias/ApplyAdam*^atrain/Adam/update_pi/l3/kernel/ApplyAdam(^atrain/Adam/update_pi/l4/bias/ApplyAdam*^atrain/Adam/update_pi/l4/kernel/ApplyAdam"[=ъ╖<ч      U'	МДПр╫AJп╬
╬ о 
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	АР
ю
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	Р
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Minimum
x"T
y"T
z"T"
Ttype:

2	Р
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
Softplus
features"T
activations"T"
Ttype:
2	
Z
SoftplusGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
1
Square
x"T
y"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И*1.9.02v1.9.0-0-g25c197e023и╨

h
statePlaceholder*
shape:         *
dtype0*'
_output_shapes
:         
Э
,critic/w1_s/Initializer/random_uniform/shapeConst*
_class
loc:@critic/w1_s*
valueB"      *
dtype0*
_output_shapes
:
П
*critic/w1_s/Initializer/random_uniform/minConst*
_class
loc:@critic/w1_s*
valueB
 *░ю╛*
dtype0*
_output_shapes
: 
П
*critic/w1_s/Initializer/random_uniform/maxConst*
_output_shapes
: *
_class
loc:@critic/w1_s*
valueB
 *░ю>*
dtype0
у
4critic/w1_s/Initializer/random_uniform/RandomUniformRandomUniform,critic/w1_s/Initializer/random_uniform/shape*
_class
loc:@critic/w1_s*
seed2 *
dtype0*
_output_shapes
:	А*

seed *
T0
╩
*critic/w1_s/Initializer/random_uniform/subSub*critic/w1_s/Initializer/random_uniform/max*critic/w1_s/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@critic/w1_s
▌
*critic/w1_s/Initializer/random_uniform/mulMul4critic/w1_s/Initializer/random_uniform/RandomUniform*critic/w1_s/Initializer/random_uniform/sub*
T0*
_class
loc:@critic/w1_s*
_output_shapes
:	А
╧
&critic/w1_s/Initializer/random_uniformAdd*critic/w1_s/Initializer/random_uniform/mul*critic/w1_s/Initializer/random_uniform/min*
T0*
_class
loc:@critic/w1_s*
_output_shapes
:	А
б
critic/w1_s
VariableV2*
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *
_class
loc:@critic/w1_s*
	container 
─
critic/w1_s/AssignAssigncritic/w1_s&critic/w1_s/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@critic/w1_s*
validate_shape(*
_output_shapes
:	А
s
critic/w1_s/readIdentitycritic/w1_s*
T0*
_class
loc:@critic/w1_s*
_output_shapes
:	А
Щ
*critic/b1/Initializer/random_uniform/shapeConst*
_class
loc:@critic/b1*
valueB"      *
dtype0*
_output_shapes
:
Л
(critic/b1/Initializer/random_uniform/minConst*
_class
loc:@critic/b1*
valueB
 *Iv╛*
dtype0*
_output_shapes
: 
Л
(critic/b1/Initializer/random_uniform/maxConst*
_class
loc:@critic/b1*
valueB
 *Iv>*
dtype0*
_output_shapes
: 
▌
2critic/b1/Initializer/random_uniform/RandomUniformRandomUniform*critic/b1/Initializer/random_uniform/shape*
T0*
_class
loc:@critic/b1*
seed2 *
dtype0*
_output_shapes
:	А*

seed 
┬
(critic/b1/Initializer/random_uniform/subSub(critic/b1/Initializer/random_uniform/max(critic/b1/Initializer/random_uniform/min*
_class
loc:@critic/b1*
_output_shapes
: *
T0
╒
(critic/b1/Initializer/random_uniform/mulMul2critic/b1/Initializer/random_uniform/RandomUniform(critic/b1/Initializer/random_uniform/sub*
_class
loc:@critic/b1*
_output_shapes
:	А*
T0
╟
$critic/b1/Initializer/random_uniformAdd(critic/b1/Initializer/random_uniform/mul(critic/b1/Initializer/random_uniform/min*
_class
loc:@critic/b1*
_output_shapes
:	А*
T0
Э
	critic/b1
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *
_class
loc:@critic/b1
╝
critic/b1/AssignAssign	critic/b1$critic/b1/Initializer/random_uniform*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0
m
critic/b1/readIdentity	critic/b1*
T0*
_class
loc:@critic/b1*
_output_shapes
:	А
Й
critic/MatMulMatMulstatecritic/w1_s/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b( 
c

critic/addAddcritic/MatMulcritic/b1/read*(
_output_shapes
:         А*
T0
R
critic/ReluRelu
critic/add*
T0*(
_output_shapes
:         А
з
1critic/l2/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@critic/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:
Щ
/critic/l2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *#
_class
loc:@critic/l2/kernel*
valueB
 *╫│▌╜*
dtype0
Щ
/critic/l2/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@critic/l2/kernel*
valueB
 *╫│▌=*
dtype0*
_output_shapes
: 
є
9critic/l2/kernel/Initializer/random_uniform/RandomUniformRandomUniform1critic/l2/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed *
T0*#
_class
loc:@critic/l2/kernel
▐
/critic/l2/kernel/Initializer/random_uniform/subSub/critic/l2/kernel/Initializer/random_uniform/max/critic/l2/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@critic/l2/kernel*
_output_shapes
: 
Є
/critic/l2/kernel/Initializer/random_uniform/mulMul9critic/l2/kernel/Initializer/random_uniform/RandomUniform/critic/l2/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АА*
T0*#
_class
loc:@critic/l2/kernel
ф
+critic/l2/kernel/Initializer/random_uniformAdd/critic/l2/kernel/Initializer/random_uniform/mul/critic/l2/kernel/Initializer/random_uniform/min* 
_output_shapes
:
АА*
T0*#
_class
loc:@critic/l2/kernel
н
critic/l2/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *#
_class
loc:@critic/l2/kernel*
	container *
shape:
АА
┘
critic/l2/kernel/AssignAssigncritic/l2/kernel+critic/l2/kernel/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@critic/l2/kernel*
validate_shape(* 
_output_shapes
:
АА
Г
critic/l2/kernel/readIdentitycritic/l2/kernel* 
_output_shapes
:
АА*
T0*#
_class
loc:@critic/l2/kernel
Т
 critic/l2/bias/Initializer/zerosConst*!
_class
loc:@critic/l2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Я
critic/l2/bias
VariableV2*!
_class
loc:@critic/l2/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
├
critic/l2/bias/AssignAssigncritic/l2/bias critic/l2/bias/Initializer/zeros*
T0*!
_class
loc:@critic/l2/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
x
critic/l2/bias/readIdentitycritic/l2/bias*
T0*!
_class
loc:@critic/l2/bias*
_output_shapes	
:А
Ч
critic/l2/MatMulMatMulcritic/Relucritic/l2/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
Н
critic/l2/BiasAddBiasAddcritic/l2/MatMulcritic/l2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
\
critic/l2/ReluRelucritic/l2/BiasAdd*
T0*(
_output_shapes
:         А
з
1critic/l3/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@critic/l3/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
Щ
/critic/l3/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *#
_class
loc:@critic/l3/kernel*
valueB
 *   ╛*
dtype0
Щ
/critic/l3/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *#
_class
loc:@critic/l3/kernel*
valueB
 *   >
є
9critic/l3/kernel/Initializer/random_uniform/RandomUniformRandomUniform1critic/l3/kernel/Initializer/random_uniform/shape*

seed *
T0*#
_class
loc:@critic/l3/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА
▐
/critic/l3/kernel/Initializer/random_uniform/subSub/critic/l3/kernel/Initializer/random_uniform/max/critic/l3/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@critic/l3/kernel*
_output_shapes
: 
Є
/critic/l3/kernel/Initializer/random_uniform/mulMul9critic/l3/kernel/Initializer/random_uniform/RandomUniform/critic/l3/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:
АА
ф
+critic/l3/kernel/Initializer/random_uniformAdd/critic/l3/kernel/Initializer/random_uniform/mul/critic/l3/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:
АА
н
critic/l3/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *#
_class
loc:@critic/l3/kernel*
	container *
shape:
АА
┘
critic/l3/kernel/AssignAssigncritic/l3/kernel+critic/l3/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*#
_class
loc:@critic/l3/kernel
Г
critic/l3/kernel/readIdentitycritic/l3/kernel*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:
АА
Т
 critic/l3/bias/Initializer/zerosConst*
_output_shapes	
:А*!
_class
loc:@critic/l3/bias*
valueBА*    *
dtype0
Я
critic/l3/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *!
_class
loc:@critic/l3/bias*
	container *
shape:А
├
critic/l3/bias/AssignAssigncritic/l3/bias critic/l3/bias/Initializer/zeros*
T0*!
_class
loc:@critic/l3/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
x
critic/l3/bias/readIdentitycritic/l3/bias*
T0*!
_class
loc:@critic/l3/bias*
_output_shapes	
:А
Ъ
critic/l3/MatMulMatMulcritic/l2/Relucritic/l3/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         А*
transpose_a( 
Н
critic/l3/BiasAddBiasAddcritic/l3/MatMulcritic/l3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
\
critic/l3/ReluRelucritic/l3/BiasAdd*(
_output_shapes
:         А*
T0
н
4critic/dense/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@critic/dense/kernel*
valueB"А      *
dtype0*
_output_shapes
:
Я
2critic/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *&
_class
loc:@critic/dense/kernel*
valueB
 *n╫\╛*
dtype0
Я
2critic/dense/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@critic/dense/kernel*
valueB
 *n╫\>*
dtype0*
_output_shapes
: 
√
<critic/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform4critic/dense/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@critic/dense/kernel*
seed2 *
dtype0*
_output_shapes
:	А
ъ
2critic/dense/kernel/Initializer/random_uniform/subSub2critic/dense/kernel/Initializer/random_uniform/max2critic/dense/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
: 
¤
2critic/dense/kernel/Initializer/random_uniform/mulMul<critic/dense/kernel/Initializer/random_uniform/RandomUniform2critic/dense/kernel/Initializer/random_uniform/sub*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	А*
T0
я
.critic/dense/kernel/Initializer/random_uniformAdd2critic/dense/kernel/Initializer/random_uniform/mul2critic/dense/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	А
▒
critic/dense/kernel
VariableV2*
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *&
_class
loc:@critic/dense/kernel*
	container 
ф
critic/dense/kernel/AssignAssigncritic/dense/kernel.critic/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*&
_class
loc:@critic/dense/kernel*
validate_shape(*
_output_shapes
:	А
Л
critic/dense/kernel/readIdentitycritic/dense/kernel*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	А*
T0
Ц
#critic/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*$
_class
loc:@critic/dense/bias*
valueB*    
г
critic/dense/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@critic/dense/bias
╬
critic/dense/bias/AssignAssigncritic/dense/bias#critic/dense/bias/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@critic/dense/bias*
validate_shape(
А
critic/dense/bias/readIdentitycritic/dense/bias*
T0*$
_class
loc:@critic/dense/bias*
_output_shapes
:
Я
critic/dense/MatMulMatMulcritic/l3/Relucritic/dense/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
Х
critic/dense/BiasAddBiasAddcritic/dense/MatMulcritic/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
v
critic/discounted_rPlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
n

critic/subSubcritic/discounted_rcritic/dense/BiasAdd*
T0*'
_output_shapes
:         
U
critic/SquareSquare
critic/sub*
T0*'
_output_shapes
:         
]
critic/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
n
critic/MeanMeancritic/Squarecritic/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Y
critic/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
_
critic/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
Д
critic/gradients/FillFillcritic/gradients/Shapecritic/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
А
/critic/gradients/critic/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
│
)critic/gradients/critic/Mean_grad/ReshapeReshapecritic/gradients/Fill/critic/gradients/critic/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
t
'critic/gradients/critic/Mean_grad/ShapeShapecritic/Square*
out_type0*
_output_shapes
:*
T0
╞
&critic/gradients/critic/Mean_grad/TileTile)critic/gradients/critic/Mean_grad/Reshape'critic/gradients/critic/Mean_grad/Shape*'
_output_shapes
:         *

Tmultiples0*
T0
v
)critic/gradients/critic/Mean_grad/Shape_1Shapecritic/Square*
T0*
out_type0*
_output_shapes
:
l
)critic/gradients/critic/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
q
'critic/gradients/critic/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
└
&critic/gradients/critic/Mean_grad/ProdProd)critic/gradients/critic/Mean_grad/Shape_1'critic/gradients/critic/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
s
)critic/gradients/critic/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
─
(critic/gradients/critic/Mean_grad/Prod_1Prod)critic/gradients/critic/Mean_grad/Shape_2)critic/gradients/critic/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
m
+critic/gradients/critic/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
м
)critic/gradients/critic/Mean_grad/MaximumMaximum(critic/gradients/critic/Mean_grad/Prod_1+critic/gradients/critic/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
к
*critic/gradients/critic/Mean_grad/floordivFloorDiv&critic/gradients/critic/Mean_grad/Prod)critic/gradients/critic/Mean_grad/Maximum*
_output_shapes
: *
T0
К
&critic/gradients/critic/Mean_grad/CastCast*critic/gradients/critic/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
╢
)critic/gradients/critic/Mean_grad/truedivRealDiv&critic/gradients/critic/Mean_grad/Tile&critic/gradients/critic/Mean_grad/Cast*'
_output_shapes
:         *
T0
Ъ
)critic/gradients/critic/Square_grad/ConstConst*^critic/gradients/critic/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
Ч
'critic/gradients/critic/Square_grad/MulMul
critic/sub)critic/gradients/critic/Square_grad/Const*'
_output_shapes
:         *
T0
╢
)critic/gradients/critic/Square_grad/Mul_1Mul)critic/gradients/critic/Mean_grad/truediv'critic/gradients/critic/Square_grad/Mul*
T0*'
_output_shapes
:         
y
&critic/gradients/critic/sub_grad/ShapeShapecritic/discounted_r*
out_type0*
_output_shapes
:*
T0
|
(critic/gradients/critic/sub_grad/Shape_1Shapecritic/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
▐
6critic/gradients/critic/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&critic/gradients/critic/sub_grad/Shape(critic/gradients/critic/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╬
$critic/gradients/critic/sub_grad/SumSum)critic/gradients/critic/Square_grad/Mul_16critic/gradients/critic/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
┴
(critic/gradients/critic/sub_grad/ReshapeReshape$critic/gradients/critic/sub_grad/Sum&critic/gradients/critic/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
╥
&critic/gradients/critic/sub_grad/Sum_1Sum)critic/gradients/critic/Square_grad/Mul_18critic/gradients/critic/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
v
$critic/gradients/critic/sub_grad/NegNeg&critic/gradients/critic/sub_grad/Sum_1*
T0*
_output_shapes
:
┼
*critic/gradients/critic/sub_grad/Reshape_1Reshape$critic/gradients/critic/sub_grad/Neg(critic/gradients/critic/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
С
1critic/gradients/critic/sub_grad/tuple/group_depsNoOp)^critic/gradients/critic/sub_grad/Reshape+^critic/gradients/critic/sub_grad/Reshape_1
Т
9critic/gradients/critic/sub_grad/tuple/control_dependencyIdentity(critic/gradients/critic/sub_grad/Reshape2^critic/gradients/critic/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@critic/gradients/critic/sub_grad/Reshape*'
_output_shapes
:         
Ш
;critic/gradients/critic/sub_grad/tuple/control_dependency_1Identity*critic/gradients/critic/sub_grad/Reshape_12^critic/gradients/critic/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@critic/gradients/critic/sub_grad/Reshape_1*'
_output_shapes
:         
╛
6critic/gradients/critic/dense/BiasAdd_grad/BiasAddGradBiasAddGrad;critic/gradients/critic/sub_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0
║
;critic/gradients/critic/dense/BiasAdd_grad/tuple/group_depsNoOp7^critic/gradients/critic/dense/BiasAdd_grad/BiasAddGrad<^critic/gradients/critic/sub_grad/tuple/control_dependency_1
╗
Ccritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependencyIdentity;critic/gradients/critic/sub_grad/tuple/control_dependency_1<^critic/gradients/critic/dense/BiasAdd_grad/tuple/group_deps*=
_class3
1/loc:@critic/gradients/critic/sub_grad/Reshape_1*'
_output_shapes
:         *
T0
╖
Ecritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependency_1Identity6critic/gradients/critic/dense/BiasAdd_grad/BiasAddGrad<^critic/gradients/critic/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*I
_class?
=;loc:@critic/gradients/critic/dense/BiasAdd_grad/BiasAddGrad
Є
0critic/gradients/critic/dense/MatMul_grad/MatMulMatMulCcritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependencycritic/dense/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b(*
T0
с
2critic/gradients/critic/dense/MatMul_grad/MatMul_1MatMulcritic/l3/ReluCcritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	А*
transpose_a(*
transpose_b( 
к
:critic/gradients/critic/dense/MatMul_grad/tuple/group_depsNoOp1^critic/gradients/critic/dense/MatMul_grad/MatMul3^critic/gradients/critic/dense/MatMul_grad/MatMul_1
╡
Bcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependencyIdentity0critic/gradients/critic/dense/MatMul_grad/MatMul;^critic/gradients/critic/dense/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@critic/gradients/critic/dense/MatMul_grad/MatMul*(
_output_shapes
:         А
▓
Dcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependency_1Identity2critic/gradients/critic/dense/MatMul_grad/MatMul_1;^critic/gradients/critic/dense/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@critic/gradients/critic/dense/MatMul_grad/MatMul_1*
_output_shapes
:	А
└
-critic/gradients/critic/l3/Relu_grad/ReluGradReluGradBcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependencycritic/l3/Relu*(
_output_shapes
:         А*
T0
о
3critic/gradients/critic/l3/BiasAdd_grad/BiasAddGradBiasAddGrad-critic/gradients/critic/l3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
ж
8critic/gradients/critic/l3/BiasAdd_grad/tuple/group_depsNoOp4^critic/gradients/critic/l3/BiasAdd_grad/BiasAddGrad.^critic/gradients/critic/l3/Relu_grad/ReluGrad
л
@critic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l3/Relu_grad/ReluGrad9^critic/gradients/critic/l3/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@critic/gradients/critic/l3/Relu_grad/ReluGrad*(
_output_shapes
:         А*
T0
м
Bcritic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependency_1Identity3critic/gradients/critic/l3/BiasAdd_grad/BiasAddGrad9^critic/gradients/critic/l3/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*F
_class<
:8loc:@critic/gradients/critic/l3/BiasAdd_grad/BiasAddGrad
щ
-critic/gradients/critic/l3/MatMul_grad/MatMulMatMul@critic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependencycritic/l3/kernel/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b(
▄
/critic/gradients/critic/l3/MatMul_grad/MatMul_1MatMulcritic/l2/Relu@critic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( *
T0
б
7critic/gradients/critic/l3/MatMul_grad/tuple/group_depsNoOp.^critic/gradients/critic/l3/MatMul_grad/MatMul0^critic/gradients/critic/l3/MatMul_grad/MatMul_1
й
?critic/gradients/critic/l3/MatMul_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l3/MatMul_grad/MatMul8^critic/gradients/critic/l3/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@critic/gradients/critic/l3/MatMul_grad/MatMul*(
_output_shapes
:         А
з
Acritic/gradients/critic/l3/MatMul_grad/tuple/control_dependency_1Identity/critic/gradients/critic/l3/MatMul_grad/MatMul_18^critic/gradients/critic/l3/MatMul_grad/tuple/group_deps* 
_output_shapes
:
АА*
T0*B
_class8
64loc:@critic/gradients/critic/l3/MatMul_grad/MatMul_1
╜
-critic/gradients/critic/l2/Relu_grad/ReluGradReluGrad?critic/gradients/critic/l3/MatMul_grad/tuple/control_dependencycritic/l2/Relu*(
_output_shapes
:         А*
T0
о
3critic/gradients/critic/l2/BiasAdd_grad/BiasAddGradBiasAddGrad-critic/gradients/critic/l2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
ж
8critic/gradients/critic/l2/BiasAdd_grad/tuple/group_depsNoOp4^critic/gradients/critic/l2/BiasAdd_grad/BiasAddGrad.^critic/gradients/critic/l2/Relu_grad/ReluGrad
л
@critic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l2/Relu_grad/ReluGrad9^critic/gradients/critic/l2/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@critic/gradients/critic/l2/Relu_grad/ReluGrad*(
_output_shapes
:         А*
T0
м
Bcritic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependency_1Identity3critic/gradients/critic/l2/BiasAdd_grad/BiasAddGrad9^critic/gradients/critic/l2/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*F
_class<
:8loc:@critic/gradients/critic/l2/BiasAdd_grad/BiasAddGrad
щ
-critic/gradients/critic/l2/MatMul_grad/MatMulMatMul@critic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependencycritic/l2/kernel/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b(
┘
/critic/gradients/critic/l2/MatMul_grad/MatMul_1MatMulcritic/Relu@critic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
АА*
transpose_a(
б
7critic/gradients/critic/l2/MatMul_grad/tuple/group_depsNoOp.^critic/gradients/critic/l2/MatMul_grad/MatMul0^critic/gradients/critic/l2/MatMul_grad/MatMul_1
й
?critic/gradients/critic/l2/MatMul_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l2/MatMul_grad/MatMul8^critic/gradients/critic/l2/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@critic/gradients/critic/l2/MatMul_grad/MatMul*(
_output_shapes
:         А
з
Acritic/gradients/critic/l2/MatMul_grad/tuple/control_dependency_1Identity/critic/gradients/critic/l2/MatMul_grad/MatMul_18^critic/gradients/critic/l2/MatMul_grad/tuple/group_deps* 
_output_shapes
:
АА*
T0*B
_class8
64loc:@critic/gradients/critic/l2/MatMul_grad/MatMul_1
╖
*critic/gradients/critic/Relu_grad/ReluGradReluGrad?critic/gradients/critic/l2/MatMul_grad/tuple/control_dependencycritic/Relu*
T0*(
_output_shapes
:         А
s
&critic/gradients/critic/add_grad/ShapeShapecritic/MatMul*
out_type0*
_output_shapes
:*
T0
y
(critic/gradients/critic/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
▐
6critic/gradients/critic/add_grad/BroadcastGradientArgsBroadcastGradientArgs&critic/gradients/critic/add_grad/Shape(critic/gradients/critic/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╧
$critic/gradients/critic/add_grad/SumSum*critic/gradients/critic/Relu_grad/ReluGrad6critic/gradients/critic/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
┬
(critic/gradients/critic/add_grad/ReshapeReshape$critic/gradients/critic/add_grad/Sum&critic/gradients/critic/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         А
╙
&critic/gradients/critic/add_grad/Sum_1Sum*critic/gradients/critic/Relu_grad/ReluGrad8critic/gradients/critic/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┐
*critic/gradients/critic/add_grad/Reshape_1Reshape&critic/gradients/critic/add_grad/Sum_1(critic/gradients/critic/add_grad/Shape_1*
_output_shapes
:	А*
T0*
Tshape0
С
1critic/gradients/critic/add_grad/tuple/group_depsNoOp)^critic/gradients/critic/add_grad/Reshape+^critic/gradients/critic/add_grad/Reshape_1
У
9critic/gradients/critic/add_grad/tuple/control_dependencyIdentity(critic/gradients/critic/add_grad/Reshape2^critic/gradients/critic/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@critic/gradients/critic/add_grad/Reshape*(
_output_shapes
:         А
Р
;critic/gradients/critic/add_grad/tuple/control_dependency_1Identity*critic/gradients/critic/add_grad/Reshape_12^critic/gradients/critic/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@critic/gradients/critic/add_grad/Reshape_1*
_output_shapes
:	А
┘
*critic/gradients/critic/MatMul_grad/MatMulMatMul9critic/gradients/critic/add_grad/tuple/control_dependencycritic/w1_s/read*'
_output_shapes
:         *
transpose_a( *
transpose_b(*
T0
╚
,critic/gradients/critic/MatMul_grad/MatMul_1MatMulstate9critic/gradients/critic/add_grad/tuple/control_dependency*
T0*
_output_shapes
:	А*
transpose_a(*
transpose_b( 
Ш
4critic/gradients/critic/MatMul_grad/tuple/group_depsNoOp+^critic/gradients/critic/MatMul_grad/MatMul-^critic/gradients/critic/MatMul_grad/MatMul_1
Ь
<critic/gradients/critic/MatMul_grad/tuple/control_dependencyIdentity*critic/gradients/critic/MatMul_grad/MatMul5^critic/gradients/critic/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@critic/gradients/critic/MatMul_grad/MatMul*'
_output_shapes
:         
Ъ
>critic/gradients/critic/MatMul_grad/tuple/control_dependency_1Identity,critic/gradients/critic/MatMul_grad/MatMul_15^critic/gradients/critic/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@critic/gradients/critic/MatMul_grad/MatMul_1*
_output_shapes
:	А
Г
 critic/beta1_power/initial_valueConst*
_class
loc:@critic/b1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Ф
critic/beta1_power
VariableV2*
shared_name *
_class
loc:@critic/b1*
	container *
shape: *
dtype0*
_output_shapes
: 
┴
critic/beta1_power/AssignAssigncritic/beta1_power critic/beta1_power/initial_value*
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
: *
use_locking(
v
critic/beta1_power/readIdentitycritic/beta1_power*
T0*
_class
loc:@critic/b1*
_output_shapes
: 
Г
 critic/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@critic/b1*
valueB
 *w╛?
Ф
critic/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@critic/b1*
	container *
shape: 
┴
critic/beta2_power/AssignAssigncritic/beta2_power critic/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@critic/b1
v
critic/beta2_power/readIdentitycritic/beta2_power*
_output_shapes
: *
T0*
_class
loc:@critic/b1
к
9critic/critic/w1_s/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
_class
loc:@critic/w1_s*
valueB"      *
dtype0
Ф
/critic/critic/w1_s/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@critic/w1_s*
valueB
 *    
∙
)critic/critic/w1_s/Adam/Initializer/zerosFill9critic/critic/w1_s/Adam/Initializer/zeros/shape_as_tensor/critic/critic/w1_s/Adam/Initializer/zeros/Const*
T0*
_class
loc:@critic/w1_s*

index_type0*
_output_shapes
:	А
н
critic/critic/w1_s/Adam
VariableV2*
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *
_class
loc:@critic/w1_s*
	container 
▀
critic/critic/w1_s/Adam/AssignAssigncritic/critic/w1_s/Adam)critic/critic/w1_s/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*
_class
loc:@critic/w1_s
Л
critic/critic/w1_s/Adam/readIdentitycritic/critic/w1_s/Adam*
T0*
_class
loc:@critic/w1_s*
_output_shapes
:	А
м
;critic/critic/w1_s/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@critic/w1_s*
valueB"      *
dtype0*
_output_shapes
:
Ц
1critic/critic/w1_s/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@critic/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
 
+critic/critic/w1_s/Adam_1/Initializer/zerosFill;critic/critic/w1_s/Adam_1/Initializer/zeros/shape_as_tensor1critic/critic/w1_s/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@critic/w1_s*

index_type0*
_output_shapes
:	А
п
critic/critic/w1_s/Adam_1
VariableV2*
shared_name *
_class
loc:@critic/w1_s*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
х
 critic/critic/w1_s/Adam_1/AssignAssigncritic/critic/w1_s/Adam_1+critic/critic/w1_s/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@critic/w1_s*
validate_shape(*
_output_shapes
:	А
П
critic/critic/w1_s/Adam_1/readIdentitycritic/critic/w1_s/Adam_1*
T0*
_class
loc:@critic/w1_s*
_output_shapes
:	А
Ь
'critic/critic/b1/Adam/Initializer/zerosConst*
_class
loc:@critic/b1*
valueB	А*    *
dtype0*
_output_shapes
:	А
й
critic/critic/b1/Adam
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *
_class
loc:@critic/b1*
	container *
shape:	А
╫
critic/critic/b1/Adam/AssignAssigncritic/critic/b1/Adam'critic/critic/b1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
:	А
Е
critic/critic/b1/Adam/readIdentitycritic/critic/b1/Adam*
T0*
_class
loc:@critic/b1*
_output_shapes
:	А
Ю
)critic/critic/b1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	А*
_class
loc:@critic/b1*
valueB	А*    
л
critic/critic/b1/Adam_1
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *
_class
loc:@critic/b1
▌
critic/critic/b1/Adam_1/AssignAssigncritic/critic/b1/Adam_1)critic/critic/b1/Adam_1/Initializer/zeros*
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
:	А*
use_locking(
Й
critic/critic/b1/Adam_1/readIdentitycritic/critic/b1/Adam_1*
T0*
_class
loc:@critic/b1*
_output_shapes
:	А
┤
>critic/critic/l2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@critic/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ю
4critic/critic/l2/kernel/Adam/Initializer/zeros/ConstConst*#
_class
loc:@critic/l2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
О
.critic/critic/l2/kernel/Adam/Initializer/zerosFill>critic/critic/l2/kernel/Adam/Initializer/zeros/shape_as_tensor4critic/critic/l2/kernel/Adam/Initializer/zeros/Const*
T0*#
_class
loc:@critic/l2/kernel*

index_type0* 
_output_shapes
:
АА
╣
critic/critic/l2/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *#
_class
loc:@critic/l2/kernel*
	container *
shape:
АА
Ї
#critic/critic/l2/kernel/Adam/AssignAssigncritic/critic/l2/kernel/Adam.critic/critic/l2/kernel/Adam/Initializer/zeros*#
_class
loc:@critic/l2/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0
Ы
!critic/critic/l2/kernel/Adam/readIdentitycritic/critic/l2/kernel/Adam*#
_class
loc:@critic/l2/kernel* 
_output_shapes
:
АА*
T0
╢
@critic/critic/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@critic/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:
а
6critic/critic/l2/kernel/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@critic/l2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
0critic/critic/l2/kernel/Adam_1/Initializer/zerosFill@critic/critic/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensor6critic/critic/l2/kernel/Adam_1/Initializer/zeros/Const*#
_class
loc:@critic/l2/kernel*

index_type0* 
_output_shapes
:
АА*
T0
╗
critic/critic/l2/kernel/Adam_1
VariableV2*
shared_name *#
_class
loc:@critic/l2/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
·
%critic/critic/l2/kernel/Adam_1/AssignAssigncritic/critic/l2/kernel/Adam_10critic/critic/l2/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*#
_class
loc:@critic/l2/kernel
Я
#critic/critic/l2/kernel/Adam_1/readIdentitycritic/critic/l2/kernel/Adam_1*
T0*#
_class
loc:@critic/l2/kernel* 
_output_shapes
:
АА
Ю
,critic/critic/l2/bias/Adam/Initializer/zerosConst*!
_class
loc:@critic/l2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
л
critic/critic/l2/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *!
_class
loc:@critic/l2/bias*
	container *
shape:А
ч
!critic/critic/l2/bias/Adam/AssignAssigncritic/critic/l2/bias/Adam,critic/critic/l2/bias/Adam/Initializer/zeros*
T0*!
_class
loc:@critic/l2/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
Р
critic/critic/l2/bias/Adam/readIdentitycritic/critic/l2/bias/Adam*
T0*!
_class
loc:@critic/l2/bias*
_output_shapes	
:А
а
.critic/critic/l2/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@critic/l2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
н
critic/critic/l2/bias/Adam_1
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *!
_class
loc:@critic/l2/bias
э
#critic/critic/l2/bias/Adam_1/AssignAssigncritic/critic/l2/bias/Adam_1.critic/critic/l2/bias/Adam_1/Initializer/zeros*
T0*!
_class
loc:@critic/l2/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
Ф
!critic/critic/l2/bias/Adam_1/readIdentitycritic/critic/l2/bias/Adam_1*
T0*!
_class
loc:@critic/l2/bias*
_output_shapes	
:А
┤
>critic/critic/l3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@critic/l3/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
Ю
4critic/critic/l3/kernel/Adam/Initializer/zeros/ConstConst*#
_class
loc:@critic/l3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
О
.critic/critic/l3/kernel/Adam/Initializer/zerosFill>critic/critic/l3/kernel/Adam/Initializer/zeros/shape_as_tensor4critic/critic/l3/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
АА*
T0*#
_class
loc:@critic/l3/kernel*

index_type0
╣
critic/critic/l3/kernel/Adam
VariableV2* 
_output_shapes
:
АА*
shared_name *#
_class
loc:@critic/l3/kernel*
	container *
shape:
АА*
dtype0
Ї
#critic/critic/l3/kernel/Adam/AssignAssigncritic/critic/l3/kernel/Adam.critic/critic/l3/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*#
_class
loc:@critic/l3/kernel
Ы
!critic/critic/l3/kernel/Adam/readIdentitycritic/critic/l3/kernel/Adam*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:
АА
╢
@critic/critic/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@critic/l3/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
а
6critic/critic/l3/kernel/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@critic/l3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
0critic/critic/l3/kernel/Adam_1/Initializer/zerosFill@critic/critic/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensor6critic/critic/l3/kernel/Adam_1/Initializer/zeros/Const*#
_class
loc:@critic/l3/kernel*

index_type0* 
_output_shapes
:
АА*
T0
╗
critic/critic/l3/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *#
_class
loc:@critic/l3/kernel*
	container *
shape:
АА
·
%critic/critic/l3/kernel/Adam_1/AssignAssigncritic/critic/l3/kernel/Adam_10critic/critic/l3/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@critic/l3/kernel*
validate_shape(* 
_output_shapes
:
АА
Я
#critic/critic/l3/kernel/Adam_1/readIdentitycritic/critic/l3/kernel/Adam_1* 
_output_shapes
:
АА*
T0*#
_class
loc:@critic/l3/kernel
Ю
,critic/critic/l3/bias/Adam/Initializer/zerosConst*!
_class
loc:@critic/l3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
л
critic/critic/l3/bias/Adam
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *!
_class
loc:@critic/l3/bias*
	container 
ч
!critic/critic/l3/bias/Adam/AssignAssigncritic/critic/l3/bias/Adam,critic/critic/l3/bias/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@critic/l3/bias*
validate_shape(*
_output_shapes	
:А
Р
critic/critic/l3/bias/Adam/readIdentitycritic/critic/l3/bias/Adam*
_output_shapes	
:А*
T0*!
_class
loc:@critic/l3/bias
а
.critic/critic/l3/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*!
_class
loc:@critic/l3/bias*
valueBА*    
н
critic/critic/l3/bias/Adam_1
VariableV2*
_output_shapes	
:А*
shared_name *!
_class
loc:@critic/l3/bias*
	container *
shape:А*
dtype0
э
#critic/critic/l3/bias/Adam_1/AssignAssigncritic/critic/l3/bias/Adam_1.critic/critic/l3/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@critic/l3/bias*
validate_shape(*
_output_shapes	
:А
Ф
!critic/critic/l3/bias/Adam_1/readIdentitycritic/critic/l3/bias/Adam_1*
T0*!
_class
loc:@critic/l3/bias*
_output_shapes	
:А
░
1critic/critic/dense/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	А*&
_class
loc:@critic/dense/kernel*
valueB	А*    
╜
critic/critic/dense/kernel/Adam
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *&
_class
loc:@critic/dense/kernel
 
&critic/critic/dense/kernel/Adam/AssignAssigncritic/critic/dense/kernel/Adam1critic/critic/dense/kernel/Adam/Initializer/zeros*&
_class
loc:@critic/dense/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0
г
$critic/critic/dense/kernel/Adam/readIdentitycritic/critic/dense/kernel/Adam*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	А
▓
3critic/critic/dense/kernel/Adam_1/Initializer/zerosConst*&
_class
loc:@critic/dense/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
┐
!critic/critic/dense/kernel/Adam_1
VariableV2*
_output_shapes
:	А*
shared_name *&
_class
loc:@critic/dense/kernel*
	container *
shape:	А*
dtype0
Е
(critic/critic/dense/kernel/Adam_1/AssignAssign!critic/critic/dense/kernel/Adam_13critic/critic/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@critic/dense/kernel*
validate_shape(*
_output_shapes
:	А
з
&critic/critic/dense/kernel/Adam_1/readIdentity!critic/critic/dense/kernel/Adam_1*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	А
в
/critic/critic/dense/bias/Adam/Initializer/zerosConst*$
_class
loc:@critic/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
п
critic/critic/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@critic/dense/bias*
	container *
shape:
Є
$critic/critic/dense/bias/Adam/AssignAssigncritic/critic/dense/bias/Adam/critic/critic/dense/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@critic/dense/bias
Ш
"critic/critic/dense/bias/Adam/readIdentitycritic/critic/dense/bias/Adam*
T0*$
_class
loc:@critic/dense/bias*
_output_shapes
:
д
1critic/critic/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*$
_class
loc:@critic/dense/bias*
valueB*    
▒
critic/critic/dense/bias/Adam_1
VariableV2*$
_class
loc:@critic/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
°
&critic/critic/dense/bias/Adam_1/AssignAssigncritic/critic/dense/bias/Adam_11critic/critic/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@critic/dense/bias
Ь
$critic/critic/dense/bias/Adam_1/readIdentitycritic/critic/dense/bias/Adam_1*
_output_shapes
:*
T0*$
_class
loc:@critic/dense/bias
^
critic/Adam/learning_rateConst*
valueB
 *╖Q9*
dtype0*
_output_shapes
: 
V
critic/Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
V
critic/Adam/beta2Const*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
X
critic/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w╠+2
п
(critic/Adam/update_critic/w1_s/ApplyAdam	ApplyAdamcritic/w1_scritic/critic/w1_s/Adamcritic/critic/w1_s/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilon>critic/gradients/critic/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	А*
use_locking( *
T0*
_class
loc:@critic/w1_s
в
&critic/Adam/update_critic/b1/ApplyAdam	ApplyAdam	critic/b1critic/critic/b1/Adamcritic/critic/b1/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilon;critic/gradients/critic/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@critic/b1*
use_nesterov( *
_output_shapes
:	А
╠
-critic/Adam/update_critic/l2/kernel/ApplyAdam	ApplyAdamcritic/l2/kernelcritic/critic/l2/kernel/Adamcritic/critic/l2/kernel/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonAcritic/gradients/critic/l2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
АА*
use_locking( *
T0*#
_class
loc:@critic/l2/kernel
╛
+critic/Adam/update_critic/l2/bias/ApplyAdam	ApplyAdamcritic/l2/biascritic/critic/l2/bias/Adamcritic/critic/l2/bias/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonBcritic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@critic/l2/bias*
use_nesterov( *
_output_shapes	
:А
╠
-critic/Adam/update_critic/l3/kernel/ApplyAdam	ApplyAdamcritic/l3/kernelcritic/critic/l3/kernel/Adamcritic/critic/l3/kernel/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonAcritic/gradients/critic/l3/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@critic/l3/kernel*
use_nesterov( * 
_output_shapes
:
АА*
use_locking( 
╛
+critic/Adam/update_critic/l3/bias/ApplyAdam	ApplyAdamcritic/l3/biascritic/critic/l3/bias/Adamcritic/critic/l3/bias/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonBcritic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependency_1*!
_class
loc:@critic/l3/bias*
use_nesterov( *
_output_shapes	
:А*
use_locking( *
T0
▌
0critic/Adam/update_critic/dense/kernel/ApplyAdam	ApplyAdamcritic/dense/kernelcritic/critic/dense/kernel/Adam!critic/critic/dense/kernel/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonDcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependency_1*
T0*&
_class
loc:@critic/dense/kernel*
use_nesterov( *
_output_shapes
:	А*
use_locking( 
╧
.critic/Adam/update_critic/dense/bias/ApplyAdam	ApplyAdamcritic/dense/biascritic/critic/dense/bias/Adamcritic/critic/dense/bias/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonEcritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@critic/dense/bias*
use_nesterov( *
_output_shapes
:
ї
critic/Adam/mulMulcritic/beta1_power/readcritic/Adam/beta1'^critic/Adam/update_critic/b1/ApplyAdam/^critic/Adam/update_critic/dense/bias/ApplyAdam1^critic/Adam/update_critic/dense/kernel/ApplyAdam,^critic/Adam/update_critic/l2/bias/ApplyAdam.^critic/Adam/update_critic/l2/kernel/ApplyAdam,^critic/Adam/update_critic/l3/bias/ApplyAdam.^critic/Adam/update_critic/l3/kernel/ApplyAdam)^critic/Adam/update_critic/w1_s/ApplyAdam*
T0*
_class
loc:@critic/b1*
_output_shapes
: 
й
critic/Adam/AssignAssigncritic/beta1_powercritic/Adam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@critic/b1
ў
critic/Adam/mul_1Mulcritic/beta2_power/readcritic/Adam/beta2'^critic/Adam/update_critic/b1/ApplyAdam/^critic/Adam/update_critic/dense/bias/ApplyAdam1^critic/Adam/update_critic/dense/kernel/ApplyAdam,^critic/Adam/update_critic/l2/bias/ApplyAdam.^critic/Adam/update_critic/l2/kernel/ApplyAdam,^critic/Adam/update_critic/l3/bias/ApplyAdam.^critic/Adam/update_critic/l3/kernel/ApplyAdam)^critic/Adam/update_critic/w1_s/ApplyAdam*
_class
loc:@critic/b1*
_output_shapes
: *
T0
н
critic/Adam/Assign_1Assigncritic/beta2_powercritic/Adam/mul_1*
use_locking( *
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
: 
│
critic/AdamNoOp^critic/Adam/Assign^critic/Adam/Assign_1'^critic/Adam/update_critic/b1/ApplyAdam/^critic/Adam/update_critic/dense/bias/ApplyAdam1^critic/Adam/update_critic/dense/kernel/ApplyAdam,^critic/Adam/update_critic/l2/bias/ApplyAdam.^critic/Adam/update_critic/l2/kernel/ApplyAdam,^critic/Adam/update_critic/l3/bias/ApplyAdam.^critic/Adam/update_critic/l3/kernel/ApplyAdam)^critic/Adam/update_critic/w1_s/ApplyAdam
Я
-pi/l1/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/l1/kernel*
valueB"      *
dtype0*
_output_shapes
:
С
+pi/l1/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/l1/kernel*
valueB
 *░ю╛*
dtype0*
_output_shapes
: 
С
+pi/l1/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/l1/kernel*
valueB
 *░ю>*
dtype0*
_output_shapes
: 
ц
5pi/l1/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l1/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@pi/l1/kernel*
seed2 *
dtype0*
_output_shapes
:	А*

seed 
╬
+pi/l1/kernel/Initializer/random_uniform/subSub+pi/l1/kernel/Initializer/random_uniform/max+pi/l1/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l1/kernel*
_output_shapes
: 
с
+pi/l1/kernel/Initializer/random_uniform/mulMul5pi/l1/kernel/Initializer/random_uniform/RandomUniform+pi/l1/kernel/Initializer/random_uniform/sub*
_output_shapes
:	А*
T0*
_class
loc:@pi/l1/kernel
╙
'pi/l1/kernel/Initializer/random_uniformAdd+pi/l1/kernel/Initializer/random_uniform/mul+pi/l1/kernel/Initializer/random_uniform/min*
_output_shapes
:	А*
T0*
_class
loc:@pi/l1/kernel
г
pi/l1/kernel
VariableV2*
shared_name *
_class
loc:@pi/l1/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
╚
pi/l1/kernel/AssignAssignpi/l1/kernel'pi/l1/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@pi/l1/kernel*
validate_shape(*
_output_shapes
:	А
v
pi/l1/kernel/readIdentitypi/l1/kernel*
_output_shapes
:	А*
T0*
_class
loc:@pi/l1/kernel
К
pi/l1/bias/Initializer/zerosConst*
_class
loc:@pi/l1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Ч

pi/l1/bias
VariableV2*
shared_name *
_class
loc:@pi/l1/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
│
pi/l1/bias/AssignAssign
pi/l1/biaspi/l1/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@pi/l1/bias
l
pi/l1/bias/readIdentity
pi/l1/bias*
_class
loc:@pi/l1/bias*
_output_shapes	
:А*
T0
Й
pi/l1/MatMulMatMulstatepi/l1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         А*
transpose_a( 
Б
pi/l1/BiasAddBiasAddpi/l1/MatMulpi/l1/bias/read*
data_formatNHWC*(
_output_shapes
:         А*
T0
T

pi/l1/ReluRelupi/l1/BiasAdd*(
_output_shapes
:         А*
T0
Я
-pi/l2/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:
С
+pi/l2/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/l2/kernel*
valueB
 *╫│▌╜*
dtype0*
_output_shapes
: 
С
+pi/l2/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/l2/kernel*
valueB
 *╫│▌=*
dtype0*
_output_shapes
: 
ч
5pi/l2/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l2/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
АА*

seed *
T0*
_class
loc:@pi/l2/kernel*
seed2 
╬
+pi/l2/kernel/Initializer/random_uniform/subSub+pi/l2/kernel/Initializer/random_uniform/max+pi/l2/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l2/kernel*
_output_shapes
: 
т
+pi/l2/kernel/Initializer/random_uniform/mulMul5pi/l2/kernel/Initializer/random_uniform/RandomUniform+pi/l2/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@pi/l2/kernel* 
_output_shapes
:
АА
╘
'pi/l2/kernel/Initializer/random_uniformAdd+pi/l2/kernel/Initializer/random_uniform/mul+pi/l2/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l2/kernel* 
_output_shapes
:
АА
е
pi/l2/kernel
VariableV2*
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *
_class
loc:@pi/l2/kernel*
	container 
╔
pi/l2/kernel/AssignAssignpi/l2/kernel'pi/l2/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@pi/l2/kernel*
validate_shape(* 
_output_shapes
:
АА
w
pi/l2/kernel/readIdentitypi/l2/kernel*
T0*
_class
loc:@pi/l2/kernel* 
_output_shapes
:
АА
К
pi/l2/bias/Initializer/zerosConst*
_class
loc:@pi/l2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Ч

pi/l2/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l2/bias*
	container *
shape:А
│
pi/l2/bias/AssignAssign
pi/l2/biaspi/l2/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l2/bias*
validate_shape(*
_output_shapes	
:А
l
pi/l2/bias/readIdentity
pi/l2/bias*
_class
loc:@pi/l2/bias*
_output_shapes	
:А*
T0
О
pi/l2/MatMulMatMul
pi/l1/Relupi/l2/kernel/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b( 
Б
pi/l2/BiasAddBiasAddpi/l2/MatMulpi/l2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
T

pi/l2/ReluRelupi/l2/BiasAdd*(
_output_shapes
:         А*
T0
Я
-pi/l3/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:
С
+pi/l3/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/l3/kernel*
valueB
 *╫│▌╜*
dtype0*
_output_shapes
: 
С
+pi/l3/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/l3/kernel*
valueB
 *╫│▌=*
dtype0*
_output_shapes
: 
ч
5pi/l3/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l3/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed *
T0*
_class
loc:@pi/l3/kernel
╬
+pi/l3/kernel/Initializer/random_uniform/subSub+pi/l3/kernel/Initializer/random_uniform/max+pi/l3/kernel/Initializer/random_uniform/min*
_class
loc:@pi/l3/kernel*
_output_shapes
: *
T0
т
+pi/l3/kernel/Initializer/random_uniform/mulMul5pi/l3/kernel/Initializer/random_uniform/RandomUniform+pi/l3/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:
АА
╘
'pi/l3/kernel/Initializer/random_uniformAdd+pi/l3/kernel/Initializer/random_uniform/mul+pi/l3/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:
АА
е
pi/l3/kernel
VariableV2*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *
_class
loc:@pi/l3/kernel
╔
pi/l3/kernel/AssignAssignpi/l3/kernel'pi/l3/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@pi/l3/kernel*
validate_shape(* 
_output_shapes
:
АА
w
pi/l3/kernel/readIdentitypi/l3/kernel*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:
АА
К
pi/l3/bias/Initializer/zerosConst*
_class
loc:@pi/l3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Ч

pi/l3/bias
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l3/bias
│
pi/l3/bias/AssignAssign
pi/l3/biaspi/l3/bias/Initializer/zeros*
T0*
_class
loc:@pi/l3/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
l
pi/l3/bias/readIdentity
pi/l3/bias*
_output_shapes	
:А*
T0*
_class
loc:@pi/l3/bias
О
pi/l3/MatMulMatMul
pi/l2/Relupi/l3/kernel/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b( 
Б
pi/l3/BiasAddBiasAddpi/l3/MatMulpi/l3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
T

pi/l3/ReluRelupi/l3/BiasAdd*
T0*(
_output_shapes
:         А
Я
-pi/l4/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/l4/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
С
+pi/l4/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/l4/kernel*
valueB
 *   ╛*
dtype0*
_output_shapes
: 
С
+pi/l4/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
_class
loc:@pi/l4/kernel*
valueB
 *   >*
dtype0
ч
5pi/l4/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l4/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed *
T0*
_class
loc:@pi/l4/kernel
╬
+pi/l4/kernel/Initializer/random_uniform/subSub+pi/l4/kernel/Initializer/random_uniform/max+pi/l4/kernel/Initializer/random_uniform/min*
_class
loc:@pi/l4/kernel*
_output_shapes
: *
T0
т
+pi/l4/kernel/Initializer/random_uniform/mulMul5pi/l4/kernel/Initializer/random_uniform/RandomUniform+pi/l4/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АА*
T0*
_class
loc:@pi/l4/kernel
╘
'pi/l4/kernel/Initializer/random_uniformAdd+pi/l4/kernel/Initializer/random_uniform/mul+pi/l4/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l4/kernel* 
_output_shapes
:
АА
е
pi/l4/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *
_class
loc:@pi/l4/kernel*
	container *
shape:
АА
╔
pi/l4/kernel/AssignAssignpi/l4/kernel'pi/l4/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@pi/l4/kernel*
validate_shape(* 
_output_shapes
:
АА
w
pi/l4/kernel/readIdentitypi/l4/kernel*
T0*
_class
loc:@pi/l4/kernel* 
_output_shapes
:
АА
К
pi/l4/bias/Initializer/zerosConst*
_class
loc:@pi/l4/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Ч

pi/l4/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l4/bias*
	container *
shape:А
│
pi/l4/bias/AssignAssign
pi/l4/biaspi/l4/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@pi/l4/bias
l
pi/l4/bias/readIdentity
pi/l4/bias*
_output_shapes	
:А*
T0*
_class
loc:@pi/l4/bias
О
pi/l4/MatMulMatMul
pi/l3/Relupi/l4/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
Б
pi/l4/BiasAddBiasAddpi/l4/MatMulpi/l4/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
T

pi/l4/ReluRelupi/l4/BiasAdd*(
_output_shapes
:         А*
T0
Э
,pi/a/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/a/kernel*
valueB"А      *
dtype0*
_output_shapes
:
П
*pi/a/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/a/kernel*
valueB
 *JQZ╛*
dtype0*
_output_shapes
: 
П
*pi/a/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/a/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
у
4pi/a/kernel/Initializer/random_uniform/RandomUniformRandomUniform,pi/a/kernel/Initializer/random_uniform/shape*
_class
loc:@pi/a/kernel*
seed2 *
dtype0*
_output_shapes
:	А*

seed *
T0
╩
*pi/a/kernel/Initializer/random_uniform/subSub*pi/a/kernel/Initializer/random_uniform/max*pi/a/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/a/kernel*
_output_shapes
: 
▌
*pi/a/kernel/Initializer/random_uniform/mulMul4pi/a/kernel/Initializer/random_uniform/RandomUniform*pi/a/kernel/Initializer/random_uniform/sub*
_output_shapes
:	А*
T0*
_class
loc:@pi/a/kernel
╧
&pi/a/kernel/Initializer/random_uniformAdd*pi/a/kernel/Initializer/random_uniform/mul*pi/a/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/a/kernel*
_output_shapes
:	А
б
pi/a/kernel
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *
_class
loc:@pi/a/kernel
─
pi/a/kernel/AssignAssignpi/a/kernel&pi/a/kernel/Initializer/random_uniform*
_class
loc:@pi/a/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0
s
pi/a/kernel/readIdentitypi/a/kernel*
_output_shapes
:	А*
T0*
_class
loc:@pi/a/kernel
Ж
pi/a/bias/Initializer/zerosConst*
_class
loc:@pi/a/bias*
valueB*    *
dtype0*
_output_shapes
:
У
	pi/a/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@pi/a/bias*
	container *
shape:
о
pi/a/bias/AssignAssign	pi/a/biaspi/a/bias/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/a/bias*
validate_shape(
h
pi/a/bias/readIdentity	pi/a/bias*
_output_shapes
:*
T0*
_class
loc:@pi/a/bias
Л
pi/a/MatMulMatMul
pi/l4/Relupi/a/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
}
pi/a/BiasAddBiasAddpi/a/MatMulpi/a/bias/read*
data_formatNHWC*'
_output_shapes
:         *
T0
Q
	pi/a/TanhTanhpi/a/BiasAdd*
T0*'
_output_shapes
:         
е
0pi/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@pi/dense/kernel*
valueB"А      
Ч
.pi/dense/kernel/Initializer/random_uniform/minConst*"
_class
loc:@pi/dense/kernel*
valueB
 *JQZ╛*
dtype0*
_output_shapes
: 
Ч
.pi/dense/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@pi/dense/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
я
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А*

seed *
T0*"
_class
loc:@pi/dense/kernel*
seed2 
┌
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@pi/dense/kernel*
_output_shapes
: *
T0
э
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	А
▀
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	А
й
pi/dense/kernel
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *"
_class
loc:@pi/dense/kernel
╘
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0

pi/dense/kernel/readIdentitypi/dense/kernel*
_output_shapes
:	А*
T0*"
_class
loc:@pi/dense/kernel
О
pi/dense/bias/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Ы
pi/dense/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@pi/dense/bias*
	container *
shape:
╛
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
t
pi/dense/bias/readIdentitypi/dense/bias*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:
У
pi/dense/MatMulMatMul
pi/l4/Relupi/dense/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
Й
pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*
data_formatNHWC*'
_output_shapes
:         *
T0
a
pi/dense/SoftplusSoftpluspi/dense/BiasAdd*
T0*'
_output_shapes
:         
V
pi/Normal/locIdentity	pi/a/Tanh*
T0*'
_output_shapes
:         
`
pi/Normal/scaleIdentitypi/dense/Softplus*'
_output_shapes
:         *
T0
е
0oldpi/l1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@oldpi/l1/kernel*
valueB"      
Ч
.oldpi/l1/kernel/Initializer/random_uniform/minConst*"
_class
loc:@oldpi/l1/kernel*
valueB
 *░ю╛*
dtype0*
_output_shapes
: 
Ч
.oldpi/l1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@oldpi/l1/kernel*
valueB
 *░ю>*
dtype0
я
8oldpi/l1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l1/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	А*

seed *
T0*"
_class
loc:@oldpi/l1/kernel
┌
.oldpi/l1/kernel/Initializer/random_uniform/subSub.oldpi/l1/kernel/Initializer/random_uniform/max.oldpi/l1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@oldpi/l1/kernel
э
.oldpi/l1/kernel/Initializer/random_uniform/mulMul8oldpi/l1/kernel/Initializer/random_uniform/RandomUniform.oldpi/l1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@oldpi/l1/kernel*
_output_shapes
:	А
▀
*oldpi/l1/kernel/Initializer/random_uniformAdd.oldpi/l1/kernel/Initializer/random_uniform/mul.oldpi/l1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l1/kernel*
_output_shapes
:	А
й
oldpi/l1/kernel
VariableV2*"
_class
loc:@oldpi/l1/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name 
╘
oldpi/l1/kernel/AssignAssignoldpi/l1/kernel*oldpi/l1/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*"
_class
loc:@oldpi/l1/kernel

oldpi/l1/kernel/readIdentityoldpi/l1/kernel*
_output_shapes
:	А*
T0*"
_class
loc:@oldpi/l1/kernel
Р
oldpi/l1/bias/Initializer/zerosConst* 
_class
loc:@oldpi/l1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Э
oldpi/l1/bias
VariableV2*
shared_name * 
_class
loc:@oldpi/l1/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
┐
oldpi/l1/bias/AssignAssignoldpi/l1/biasoldpi/l1/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@oldpi/l1/bias*
validate_shape(*
_output_shapes	
:А
u
oldpi/l1/bias/readIdentityoldpi/l1/bias*
T0* 
_class
loc:@oldpi/l1/bias*
_output_shapes	
:А
П
oldpi/l1/MatMulMatMulstateoldpi/l1/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
К
oldpi/l1/BiasAddBiasAddoldpi/l1/MatMuloldpi/l1/bias/read*
data_formatNHWC*(
_output_shapes
:         А*
T0
Z
oldpi/l1/ReluReluoldpi/l1/BiasAdd*
T0*(
_output_shapes
:         А
е
0oldpi/l2/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@oldpi/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ч
.oldpi/l2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *"
_class
loc:@oldpi/l2/kernel*
valueB
 *╫│▌╜*
dtype0
Ч
.oldpi/l2/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@oldpi/l2/kernel*
valueB
 *╫│▌=*
dtype0*
_output_shapes
: 
Ё
8oldpi/l2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l2/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@oldpi/l2/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed 
┌
.oldpi/l2/kernel/Initializer/random_uniform/subSub.oldpi/l2/kernel/Initializer/random_uniform/max.oldpi/l2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@oldpi/l2/kernel
ю
.oldpi/l2/kernel/Initializer/random_uniform/mulMul8oldpi/l2/kernel/Initializer/random_uniform/RandomUniform.oldpi/l2/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АА*
T0*"
_class
loc:@oldpi/l2/kernel
р
*oldpi/l2/kernel/Initializer/random_uniformAdd.oldpi/l2/kernel/Initializer/random_uniform/mul.oldpi/l2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l2/kernel* 
_output_shapes
:
АА
л
oldpi/l2/kernel
VariableV2*"
_class
loc:@oldpi/l2/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name 
╒
oldpi/l2/kernel/AssignAssignoldpi/l2/kernel*oldpi/l2/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@oldpi/l2/kernel*
validate_shape(* 
_output_shapes
:
АА
А
oldpi/l2/kernel/readIdentityoldpi/l2/kernel* 
_output_shapes
:
АА*
T0*"
_class
loc:@oldpi/l2/kernel
Р
oldpi/l2/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А* 
_class
loc:@oldpi/l2/bias*
valueBА*    
Э
oldpi/l2/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name * 
_class
loc:@oldpi/l2/bias*
	container *
shape:А
┐
oldpi/l2/bias/AssignAssignoldpi/l2/biasoldpi/l2/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@oldpi/l2/bias*
validate_shape(*
_output_shapes	
:А
u
oldpi/l2/bias/readIdentityoldpi/l2/bias*
_output_shapes	
:А*
T0* 
_class
loc:@oldpi/l2/bias
Ч
oldpi/l2/MatMulMatMuloldpi/l1/Reluoldpi/l2/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
К
oldpi/l2/BiasAddBiasAddoldpi/l2/MatMuloldpi/l2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
Z
oldpi/l2/ReluReluoldpi/l2/BiasAdd*
T0*(
_output_shapes
:         А
е
0oldpi/l3/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@oldpi/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ч
.oldpi/l3/kernel/Initializer/random_uniform/minConst*"
_class
loc:@oldpi/l3/kernel*
valueB
 *╫│▌╜*
dtype0*
_output_shapes
: 
Ч
.oldpi/l3/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@oldpi/l3/kernel*
valueB
 *╫│▌=*
dtype0*
_output_shapes
: 
Ё
8oldpi/l3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l3/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@oldpi/l3/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed 
┌
.oldpi/l3/kernel/Initializer/random_uniform/subSub.oldpi/l3/kernel/Initializer/random_uniform/max.oldpi/l3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l3/kernel*
_output_shapes
: 
ю
.oldpi/l3/kernel/Initializer/random_uniform/mulMul8oldpi/l3/kernel/Initializer/random_uniform/RandomUniform.oldpi/l3/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@oldpi/l3/kernel* 
_output_shapes
:
АА
р
*oldpi/l3/kernel/Initializer/random_uniformAdd.oldpi/l3/kernel/Initializer/random_uniform/mul.oldpi/l3/kernel/Initializer/random_uniform/min* 
_output_shapes
:
АА*
T0*"
_class
loc:@oldpi/l3/kernel
л
oldpi/l3/kernel
VariableV2*"
_class
loc:@oldpi/l3/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name 
╒
oldpi/l3/kernel/AssignAssignoldpi/l3/kernel*oldpi/l3/kernel/Initializer/random_uniform*
T0*"
_class
loc:@oldpi/l3/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
А
oldpi/l3/kernel/readIdentityoldpi/l3/kernel*
T0*"
_class
loc:@oldpi/l3/kernel* 
_output_shapes
:
АА
Р
oldpi/l3/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А* 
_class
loc:@oldpi/l3/bias*
valueBА*    
Э
oldpi/l3/bias
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name * 
_class
loc:@oldpi/l3/bias*
	container 
┐
oldpi/l3/bias/AssignAssignoldpi/l3/biasoldpi/l3/bias/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0* 
_class
loc:@oldpi/l3/bias*
validate_shape(
u
oldpi/l3/bias/readIdentityoldpi/l3/bias* 
_class
loc:@oldpi/l3/bias*
_output_shapes	
:А*
T0
Ч
oldpi/l3/MatMulMatMuloldpi/l2/Reluoldpi/l3/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
К
oldpi/l3/BiasAddBiasAddoldpi/l3/MatMuloldpi/l3/bias/read*
data_formatNHWC*(
_output_shapes
:         А*
T0
Z
oldpi/l3/ReluReluoldpi/l3/BiasAdd*
T0*(
_output_shapes
:         А
е
0oldpi/l4/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*"
_class
loc:@oldpi/l4/kernel*
valueB"   А   *
dtype0
Ч
.oldpi/l4/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@oldpi/l4/kernel*
valueB
 *   ╛
Ч
.oldpi/l4/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@oldpi/l4/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
Ё
8oldpi/l4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l4/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
АА*

seed *
T0*"
_class
loc:@oldpi/l4/kernel*
seed2 
┌
.oldpi/l4/kernel/Initializer/random_uniform/subSub.oldpi/l4/kernel/Initializer/random_uniform/max.oldpi/l4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l4/kernel*
_output_shapes
: 
ю
.oldpi/l4/kernel/Initializer/random_uniform/mulMul8oldpi/l4/kernel/Initializer/random_uniform/RandomUniform.oldpi/l4/kernel/Initializer/random_uniform/sub*"
_class
loc:@oldpi/l4/kernel* 
_output_shapes
:
АА*
T0
р
*oldpi/l4/kernel/Initializer/random_uniformAdd.oldpi/l4/kernel/Initializer/random_uniform/mul.oldpi/l4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l4/kernel* 
_output_shapes
:
АА
л
oldpi/l4/kernel
VariableV2*
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *"
_class
loc:@oldpi/l4/kernel*
	container 
╒
oldpi/l4/kernel/AssignAssignoldpi/l4/kernel*oldpi/l4/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*"
_class
loc:@oldpi/l4/kernel
А
oldpi/l4/kernel/readIdentityoldpi/l4/kernel* 
_output_shapes
:
АА*
T0*"
_class
loc:@oldpi/l4/kernel
Р
oldpi/l4/bias/Initializer/zerosConst* 
_class
loc:@oldpi/l4/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Э
oldpi/l4/bias
VariableV2*
shared_name * 
_class
loc:@oldpi/l4/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
┐
oldpi/l4/bias/AssignAssignoldpi/l4/biasoldpi/l4/bias/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0* 
_class
loc:@oldpi/l4/bias*
validate_shape(
u
oldpi/l4/bias/readIdentityoldpi/l4/bias*
T0* 
_class
loc:@oldpi/l4/bias*
_output_shapes	
:А
Ч
oldpi/l4/MatMulMatMuloldpi/l3/Reluoldpi/l4/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
К
oldpi/l4/BiasAddBiasAddoldpi/l4/MatMuloldpi/l4/bias/read*
data_formatNHWC*(
_output_shapes
:         А*
T0
Z
oldpi/l4/ReluReluoldpi/l4/BiasAdd*
T0*(
_output_shapes
:         А
г
/oldpi/a/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@oldpi/a/kernel*
valueB"А      *
dtype0*
_output_shapes
:
Х
-oldpi/a/kernel/Initializer/random_uniform/minConst*!
_class
loc:@oldpi/a/kernel*
valueB
 *JQZ╛*
dtype0*
_output_shapes
: 
Х
-oldpi/a/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *!
_class
loc:@oldpi/a/kernel*
valueB
 *JQZ>
ь
7oldpi/a/kernel/Initializer/random_uniform/RandomUniformRandomUniform/oldpi/a/kernel/Initializer/random_uniform/shape*
_output_shapes
:	А*

seed *
T0*!
_class
loc:@oldpi/a/kernel*
seed2 *
dtype0
╓
-oldpi/a/kernel/Initializer/random_uniform/subSub-oldpi/a/kernel/Initializer/random_uniform/max-oldpi/a/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@oldpi/a/kernel*
_output_shapes
: 
щ
-oldpi/a/kernel/Initializer/random_uniform/mulMul7oldpi/a/kernel/Initializer/random_uniform/RandomUniform-oldpi/a/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@oldpi/a/kernel*
_output_shapes
:	А
█
)oldpi/a/kernel/Initializer/random_uniformAdd-oldpi/a/kernel/Initializer/random_uniform/mul-oldpi/a/kernel/Initializer/random_uniform/min*!
_class
loc:@oldpi/a/kernel*
_output_shapes
:	А*
T0
з
oldpi/a/kernel
VariableV2*!
_class
loc:@oldpi/a/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name 
╨
oldpi/a/kernel/AssignAssignoldpi/a/kernel)oldpi/a/kernel/Initializer/random_uniform*
T0*!
_class
loc:@oldpi/a/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(
|
oldpi/a/kernel/readIdentityoldpi/a/kernel*
T0*!
_class
loc:@oldpi/a/kernel*
_output_shapes
:	А
М
oldpi/a/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@oldpi/a/bias*
valueB*    
Щ
oldpi/a/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@oldpi/a/bias
║
oldpi/a/bias/AssignAssignoldpi/a/biasoldpi/a/bias/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@oldpi/a/bias*
validate_shape(
q
oldpi/a/bias/readIdentityoldpi/a/bias*
T0*
_class
loc:@oldpi/a/bias*
_output_shapes
:
Ф
oldpi/a/MatMulMatMuloldpi/l4/Reluoldpi/a/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
Ж
oldpi/a/BiasAddBiasAddoldpi/a/MatMuloldpi/a/bias/read*'
_output_shapes
:         *
T0*
data_formatNHWC
W
oldpi/a/TanhTanholdpi/a/BiasAdd*
T0*'
_output_shapes
:         
л
3oldpi/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
_class
loc:@oldpi/dense/kernel*
valueB"А      *
dtype0
Э
1oldpi/dense/kernel/Initializer/random_uniform/minConst*%
_class
loc:@oldpi/dense/kernel*
valueB
 *JQZ╛*
dtype0*
_output_shapes
: 
Э
1oldpi/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *%
_class
loc:@oldpi/dense/kernel*
valueB
 *JQZ>
°
;oldpi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform3oldpi/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А*

seed *
T0*%
_class
loc:@oldpi/dense/kernel*
seed2 
ц
1oldpi/dense/kernel/Initializer/random_uniform/subSub1oldpi/dense/kernel/Initializer/random_uniform/max1oldpi/dense/kernel/Initializer/random_uniform/min*
T0*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
: 
∙
1oldpi/dense/kernel/Initializer/random_uniform/mulMul;oldpi/dense/kernel/Initializer/random_uniform/RandomUniform1oldpi/dense/kernel/Initializer/random_uniform/sub*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
:	А*
T0
ы
-oldpi/dense/kernel/Initializer/random_uniformAdd1oldpi/dense/kernel/Initializer/random_uniform/mul1oldpi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	А*
T0*%
_class
loc:@oldpi/dense/kernel
п
oldpi/dense/kernel
VariableV2*
shared_name *%
_class
loc:@oldpi/dense/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
р
oldpi/dense/kernel/AssignAssignoldpi/dense/kernel-oldpi/dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*%
_class
loc:@oldpi/dense/kernel
И
oldpi/dense/kernel/readIdentityoldpi/dense/kernel*
T0*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
:	А
Ф
"oldpi/dense/bias/Initializer/zerosConst*#
_class
loc:@oldpi/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
б
oldpi/dense/bias
VariableV2*
shared_name *#
_class
loc:@oldpi/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
╩
oldpi/dense/bias/AssignAssignoldpi/dense/bias"oldpi/dense/bias/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@oldpi/dense/bias*
validate_shape(*
_output_shapes
:
}
oldpi/dense/bias/readIdentityoldpi/dense/bias*
_output_shapes
:*
T0*#
_class
loc:@oldpi/dense/bias
Ь
oldpi/dense/MatMulMatMuloldpi/l4/Reluoldpi/dense/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
Т
oldpi/dense/BiasAddBiasAddoldpi/dense/MatMuloldpi/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
g
oldpi/dense/SoftplusSoftplusoldpi/dense/BiasAdd*
T0*'
_output_shapes
:         
\
oldpi/Normal/locIdentityoldpi/a/Tanh*
T0*'
_output_shapes
:         
f
oldpi/Normal/scaleIdentityoldpi/dense/Softplus*'
_output_shapes
:         *
T0
_
pi/Normal/sample/sample_shapeConst*
value	B :*
dtype0*
_output_shapes
: 
i
pi/Normal/sample/sample_shape_1Const*
dtype0*
_output_shapes
:*
valueB:
o
"pi/Normal/batch_shape_tensor/ShapeShapepi/Normal/loc*
T0*
out_type0*
_output_shapes
:
s
$pi/Normal/batch_shape_tensor/Shape_1Shapepi/Normal/scale*
T0*
out_type0*
_output_shapes
:
к
*pi/Normal/batch_shape_tensor/BroadcastArgsBroadcastArgs"pi/Normal/batch_shape_tensor/Shape$pi/Normal/batch_shape_tensor/Shape_1*
_output_shapes
:*
T0
j
 pi/Normal/sample/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
^
pi/Normal/sample/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╔
pi/Normal/sample/concatConcatV2 pi/Normal/sample/concat/values_0*pi/Normal/batch_shape_tensor/BroadcastArgspi/Normal/sample/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
h
#pi/Normal/sample/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
j
%pi/Normal/sample/random_normal/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╔
3pi/Normal/sample/random_normal/RandomStandardNormalRandomStandardNormalpi/Normal/sample/concat*
T0*
dtype0*4
_output_shapes"
 :                  *
seed2 *

seed 
─
"pi/Normal/sample/random_normal/mulMul3pi/Normal/sample/random_normal/RandomStandardNormal%pi/Normal/sample/random_normal/stddev*
T0*4
_output_shapes"
 :                  
н
pi/Normal/sample/random_normalAdd"pi/Normal/sample/random_normal/mul#pi/Normal/sample/random_normal/mean*4
_output_shapes"
 :                  *
T0
В
pi/Normal/sample/mulMulpi/Normal/sample/random_normalpi/Normal/scale*
T0*+
_output_shapes
:         
v
pi/Normal/sample/addAddpi/Normal/sample/mulpi/Normal/loc*+
_output_shapes
:         *
T0
j
pi/Normal/sample/ShapeShapepi/Normal/sample/add*
T0*
out_type0*
_output_shapes
:
n
$pi/Normal/sample/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
p
&pi/Normal/sample/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
p
&pi/Normal/sample/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╥
pi/Normal/sample/strided_sliceStridedSlicepi/Normal/sample/Shape$pi/Normal/sample/strided_slice/stack&pi/Normal/sample/strided_slice/stack_1&pi/Normal/sample/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
T0*
Index0
`
pi/Normal/sample/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
└
pi/Normal/sample/concat_1ConcatV2pi/Normal/sample/sample_shape_1pi/Normal/sample/strided_slicepi/Normal/sample/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ш
pi/Normal/sample/ReshapeReshapepi/Normal/sample/addpi/Normal/sample/concat_1*+
_output_shapes
:         *
T0*
Tshape0
Г
sample_action/SqueezeSqueezepi/Normal/sample/Reshape*
T0*'
_output_shapes
:         *
squeeze_dims
 
╕
update_oldpi/AssignAssignoldpi/l1/kernelpi/l1/kernel/read*
use_locking( *
T0*"
_class
loc:@oldpi/l1/kernel*
validate_shape(*
_output_shapes
:	А
░
update_oldpi/Assign_1Assignoldpi/l1/biaspi/l1/bias/read*
validate_shape(*
_output_shapes	
:А*
use_locking( *
T0* 
_class
loc:@oldpi/l1/bias
╗
update_oldpi/Assign_2Assignoldpi/l2/kernelpi/l2/kernel/read*
T0*"
_class
loc:@oldpi/l2/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking( 
░
update_oldpi/Assign_3Assignoldpi/l2/biaspi/l2/bias/read*
use_locking( *
T0* 
_class
loc:@oldpi/l2/bias*
validate_shape(*
_output_shapes	
:А
╗
update_oldpi/Assign_4Assignoldpi/l3/kernelpi/l3/kernel/read*
use_locking( *
T0*"
_class
loc:@oldpi/l3/kernel*
validate_shape(* 
_output_shapes
:
АА
░
update_oldpi/Assign_5Assignoldpi/l3/biaspi/l3/bias/read*
T0* 
_class
loc:@oldpi/l3/bias*
validate_shape(*
_output_shapes	
:А*
use_locking( 
╗
update_oldpi/Assign_6Assignoldpi/l4/kernelpi/l4/kernel/read*
use_locking( *
T0*"
_class
loc:@oldpi/l4/kernel*
validate_shape(* 
_output_shapes
:
АА
░
update_oldpi/Assign_7Assignoldpi/l4/biaspi/l4/bias/read*
_output_shapes	
:А*
use_locking( *
T0* 
_class
loc:@oldpi/l4/bias*
validate_shape(
╖
update_oldpi/Assign_8Assignoldpi/a/kernelpi/a/kernel/read*
T0*!
_class
loc:@oldpi/a/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking( 
м
update_oldpi/Assign_9Assignoldpi/a/biaspi/a/bias/read*
T0*
_class
loc:@oldpi/a/bias*
validate_shape(*
_output_shapes
:*
use_locking( 
─
update_oldpi/Assign_10Assignoldpi/dense/kernelpi/dense/kernel/read*%
_class
loc:@oldpi/dense/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking( *
T0
╣
update_oldpi/Assign_11Assignoldpi/dense/biaspi/dense/bias/read*
T0*#
_class
loc:@oldpi/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking( 
i
actionPlaceholder*
shape:         *
dtype0*'
_output_shapes
:         
l
	advantagePlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
n
pi/Normal/prob/standardize/subSubactionpi/Normal/loc*'
_output_shapes
:         *
T0
Р
"pi/Normal/prob/standardize/truedivRealDivpi/Normal/prob/standardize/subpi/Normal/scale*'
_output_shapes
:         *
T0
u
pi/Normal/prob/SquareSquare"pi/Normal/prob/standardize/truediv*
T0*'
_output_shapes
:         
Y
pi/Normal/prob/mul/xConst*
valueB
 *   ┐*
dtype0*
_output_shapes
: 
x
pi/Normal/prob/mulMulpi/Normal/prob/mul/xpi/Normal/prob/Square*'
_output_shapes
:         *
T0
\
pi/Normal/prob/LogLogpi/Normal/scale*'
_output_shapes
:         *
T0
Y
pi/Normal/prob/add/xConst*
valueB
 *О?k?*
dtype0*
_output_shapes
: 
u
pi/Normal/prob/addAddpi/Normal/prob/add/xpi/Normal/prob/Log*
T0*'
_output_shapes
:         
s
pi/Normal/prob/subSubpi/Normal/prob/mulpi/Normal/prob/add*
T0*'
_output_shapes
:         
_
pi/Normal/prob/ExpExppi/Normal/prob/sub*'
_output_shapes
:         *
T0
t
!oldpi/Normal/prob/standardize/subSubactionoldpi/Normal/loc*'
_output_shapes
:         *
T0
Щ
%oldpi/Normal/prob/standardize/truedivRealDiv!oldpi/Normal/prob/standardize/suboldpi/Normal/scale*
T0*'
_output_shapes
:         
{
oldpi/Normal/prob/SquareSquare%oldpi/Normal/prob/standardize/truediv*
T0*'
_output_shapes
:         
\
oldpi/Normal/prob/mul/xConst*
_output_shapes
: *
valueB
 *   ┐*
dtype0
Б
oldpi/Normal/prob/mulMuloldpi/Normal/prob/mul/xoldpi/Normal/prob/Square*
T0*'
_output_shapes
:         
b
oldpi/Normal/prob/LogLogoldpi/Normal/scale*'
_output_shapes
:         *
T0
\
oldpi/Normal/prob/add/xConst*
valueB
 *О?k?*
dtype0*
_output_shapes
: 
~
oldpi/Normal/prob/addAddoldpi/Normal/prob/add/xoldpi/Normal/prob/Log*'
_output_shapes
:         *
T0
|
oldpi/Normal/prob/subSuboldpi/Normal/prob/muloldpi/Normal/prob/add*
T0*'
_output_shapes
:         
e
oldpi/Normal/prob/ExpExpoldpi/Normal/prob/sub*
T0*'
_output_shapes
:         
~
loss/surrogate/truedivRealDivpi/Normal/prob/Expoldpi/Normal/prob/Exp*'
_output_shapes
:         *
T0
n
loss/surrogate/mulMulloss/surrogate/truediv	advantage*
T0*'
_output_shapes
:         
a
loss/clip_by_value/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *ЪЩЩ?
Н
loss/clip_by_value/MinimumMinimumloss/surrogate/truedivloss/clip_by_value/Minimum/y*'
_output_shapes
:         *
T0
Y
loss/clip_by_value/yConst*
dtype0*
_output_shapes
: *
valueB
 *═╠L?
Б
loss/clip_by_valueMaximumloss/clip_by_value/Minimumloss/clip_by_value/y*'
_output_shapes
:         *
T0
`
loss/mulMulloss/clip_by_value	advantage*
T0*'
_output_shapes
:         
g
loss/MinimumMinimumloss/surrogate/mulloss/mul*
T0*'
_output_shapes
:         
[

loss/ConstConst*
_output_shapes
:*
valueB"       *
dtype0
i
	loss/MeanMeanloss/Minimum
loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
;
loss/NegNeg	loss/Mean*
T0*
_output_shapes
: 
Y
atrain/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
_
atrain/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Д
atrain/gradients/FillFillatrain/gradients/Shapeatrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
a
"atrain/gradients/loss/Neg_grad/NegNegatrain/gradients/Fill*
_output_shapes
: *
T0
~
-atrain/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
╝
'atrain/gradients/loss/Mean_grad/ReshapeReshape"atrain/gradients/loss/Neg_grad/Neg-atrain/gradients/loss/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
q
%atrain/gradients/loss/Mean_grad/ShapeShapeloss/Minimum*
T0*
out_type0*
_output_shapes
:
└
$atrain/gradients/loss/Mean_grad/TileTile'atrain/gradients/loss/Mean_grad/Reshape%atrain/gradients/loss/Mean_grad/Shape*'
_output_shapes
:         *

Tmultiples0*
T0
s
'atrain/gradients/loss/Mean_grad/Shape_1Shapeloss/Minimum*
T0*
out_type0*
_output_shapes
:
j
'atrain/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
o
%atrain/gradients/loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
║
$atrain/gradients/loss/Mean_grad/ProdProd'atrain/gradients/loss/Mean_grad/Shape_1%atrain/gradients/loss/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
q
'atrain/gradients/loss/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
╛
&atrain/gradients/loss/Mean_grad/Prod_1Prod'atrain/gradients/loss/Mean_grad/Shape_2'atrain/gradients/loss/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
k
)atrain/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
ж
'atrain/gradients/loss/Mean_grad/MaximumMaximum&atrain/gradients/loss/Mean_grad/Prod_1)atrain/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
д
(atrain/gradients/loss/Mean_grad/floordivFloorDiv$atrain/gradients/loss/Mean_grad/Prod'atrain/gradients/loss/Mean_grad/Maximum*
_output_shapes
: *
T0
Ж
$atrain/gradients/loss/Mean_grad/CastCast(atrain/gradients/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
░
'atrain/gradients/loss/Mean_grad/truedivRealDiv$atrain/gradients/loss/Mean_grad/Tile$atrain/gradients/loss/Mean_grad/Cast*
T0*'
_output_shapes
:         
z
(atrain/gradients/loss/Minimum_grad/ShapeShapeloss/surrogate/mul*
T0*
out_type0*
_output_shapes
:
r
*atrain/gradients/loss/Minimum_grad/Shape_1Shapeloss/mul*
T0*
out_type0*
_output_shapes
:
С
*atrain/gradients/loss/Minimum_grad/Shape_2Shape'atrain/gradients/loss/Mean_grad/truediv*
T0*
out_type0*
_output_shapes
:
s
.atrain/gradients/loss/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╨
(atrain/gradients/loss/Minimum_grad/zerosFill*atrain/gradients/loss/Minimum_grad/Shape_2.atrain/gradients/loss/Minimum_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:         
Й
,atrain/gradients/loss/Minimum_grad/LessEqual	LessEqualloss/surrogate/mulloss/mul*'
_output_shapes
:         *
T0
ф
8atrain/gradients/loss/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs(atrain/gradients/loss/Minimum_grad/Shape*atrain/gradients/loss/Minimum_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ц
)atrain/gradients/loss/Minimum_grad/SelectSelect,atrain/gradients/loss/Minimum_grad/LessEqual'atrain/gradients/loss/Mean_grad/truediv(atrain/gradients/loss/Minimum_grad/zeros*
T0*'
_output_shapes
:         
ш
+atrain/gradients/loss/Minimum_grad/Select_1Select,atrain/gradients/loss/Minimum_grad/LessEqual(atrain/gradients/loss/Minimum_grad/zeros'atrain/gradients/loss/Mean_grad/truediv*
T0*'
_output_shapes
:         
╥
&atrain/gradients/loss/Minimum_grad/SumSum)atrain/gradients/loss/Minimum_grad/Select8atrain/gradients/loss/Minimum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╟
*atrain/gradients/loss/Minimum_grad/ReshapeReshape&atrain/gradients/loss/Minimum_grad/Sum(atrain/gradients/loss/Minimum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
╪
(atrain/gradients/loss/Minimum_grad/Sum_1Sum+atrain/gradients/loss/Minimum_grad/Select_1:atrain/gradients/loss/Minimum_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
═
,atrain/gradients/loss/Minimum_grad/Reshape_1Reshape(atrain/gradients/loss/Minimum_grad/Sum_1*atrain/gradients/loss/Minimum_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
Ч
3atrain/gradients/loss/Minimum_grad/tuple/group_depsNoOp+^atrain/gradients/loss/Minimum_grad/Reshape-^atrain/gradients/loss/Minimum_grad/Reshape_1
Ъ
;atrain/gradients/loss/Minimum_grad/tuple/control_dependencyIdentity*atrain/gradients/loss/Minimum_grad/Reshape4^atrain/gradients/loss/Minimum_grad/tuple/group_deps*
T0*=
_class3
1/loc:@atrain/gradients/loss/Minimum_grad/Reshape*'
_output_shapes
:         
а
=atrain/gradients/loss/Minimum_grad/tuple/control_dependency_1Identity,atrain/gradients/loss/Minimum_grad/Reshape_14^atrain/gradients/loss/Minimum_grad/tuple/group_deps*
T0*?
_class5
31loc:@atrain/gradients/loss/Minimum_grad/Reshape_1*'
_output_shapes
:         
Д
.atrain/gradients/loss/surrogate/mul_grad/ShapeShapeloss/surrogate/truediv*
T0*
out_type0*
_output_shapes
:
y
0atrain/gradients/loss/surrogate/mul_grad/Shape_1Shape	advantage*
T0*
out_type0*
_output_shapes
:
Ў
>atrain/gradients/loss/surrogate/mul_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/loss/surrogate/mul_grad/Shape0atrain/gradients/loss/surrogate/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
н
,atrain/gradients/loss/surrogate/mul_grad/MulMul;atrain/gradients/loss/Minimum_grad/tuple/control_dependency	advantage*
T0*'
_output_shapes
:         
с
,atrain/gradients/loss/surrogate/mul_grad/SumSum,atrain/gradients/loss/surrogate/mul_grad/Mul>atrain/gradients/loss/surrogate/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
┘
0atrain/gradients/loss/surrogate/mul_grad/ReshapeReshape,atrain/gradients/loss/surrogate/mul_grad/Sum.atrain/gradients/loss/surrogate/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
╝
.atrain/gradients/loss/surrogate/mul_grad/Mul_1Mulloss/surrogate/truediv;atrain/gradients/loss/Minimum_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
ч
.atrain/gradients/loss/surrogate/mul_grad/Sum_1Sum.atrain/gradients/loss/surrogate/mul_grad/Mul_1@atrain/gradients/loss/surrogate/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
▀
2atrain/gradients/loss/surrogate/mul_grad/Reshape_1Reshape.atrain/gradients/loss/surrogate/mul_grad/Sum_10atrain/gradients/loss/surrogate/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
й
9atrain/gradients/loss/surrogate/mul_grad/tuple/group_depsNoOp1^atrain/gradients/loss/surrogate/mul_grad/Reshape3^atrain/gradients/loss/surrogate/mul_grad/Reshape_1
▓
Aatrain/gradients/loss/surrogate/mul_grad/tuple/control_dependencyIdentity0atrain/gradients/loss/surrogate/mul_grad/Reshape:^atrain/gradients/loss/surrogate/mul_grad/tuple/group_deps*'
_output_shapes
:         *
T0*C
_class9
75loc:@atrain/gradients/loss/surrogate/mul_grad/Reshape
╕
Catrain/gradients/loss/surrogate/mul_grad/tuple/control_dependency_1Identity2atrain/gradients/loss/surrogate/mul_grad/Reshape_1:^atrain/gradients/loss/surrogate/mul_grad/tuple/group_deps*'
_output_shapes
:         *
T0*E
_class;
97loc:@atrain/gradients/loss/surrogate/mul_grad/Reshape_1
v
$atrain/gradients/loss/mul_grad/ShapeShapeloss/clip_by_value*
out_type0*
_output_shapes
:*
T0
o
&atrain/gradients/loss/mul_grad/Shape_1Shape	advantage*
T0*
out_type0*
_output_shapes
:
╪
4atrain/gradients/loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs$atrain/gradients/loss/mul_grad/Shape&atrain/gradients/loss/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
е
"atrain/gradients/loss/mul_grad/MulMul=atrain/gradients/loss/Minimum_grad/tuple/control_dependency_1	advantage*'
_output_shapes
:         *
T0
├
"atrain/gradients/loss/mul_grad/SumSum"atrain/gradients/loss/mul_grad/Mul4atrain/gradients/loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╗
&atrain/gradients/loss/mul_grad/ReshapeReshape"atrain/gradients/loss/mul_grad/Sum$atrain/gradients/loss/mul_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
░
$atrain/gradients/loss/mul_grad/Mul_1Mulloss/clip_by_value=atrain/gradients/loss/Minimum_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
╔
$atrain/gradients/loss/mul_grad/Sum_1Sum$atrain/gradients/loss/mul_grad/Mul_16atrain/gradients/loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
┴
(atrain/gradients/loss/mul_grad/Reshape_1Reshape$atrain/gradients/loss/mul_grad/Sum_1&atrain/gradients/loss/mul_grad/Shape_1*
Tshape0*'
_output_shapes
:         *
T0
Л
/atrain/gradients/loss/mul_grad/tuple/group_depsNoOp'^atrain/gradients/loss/mul_grad/Reshape)^atrain/gradients/loss/mul_grad/Reshape_1
К
7atrain/gradients/loss/mul_grad/tuple/control_dependencyIdentity&atrain/gradients/loss/mul_grad/Reshape0^atrain/gradients/loss/mul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@atrain/gradients/loss/mul_grad/Reshape*'
_output_shapes
:         
Р
9atrain/gradients/loss/mul_grad/tuple/control_dependency_1Identity(atrain/gradients/loss/mul_grad/Reshape_10^atrain/gradients/loss/mul_grad/tuple/group_deps*;
_class1
/-loc:@atrain/gradients/loss/mul_grad/Reshape_1*'
_output_shapes
:         *
T0
И
.atrain/gradients/loss/clip_by_value_grad/ShapeShapeloss/clip_by_value/Minimum*
T0*
out_type0*
_output_shapes
:
s
0atrain/gradients/loss/clip_by_value_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
з
0atrain/gradients/loss/clip_by_value_grad/Shape_2Shape7atrain/gradients/loss/mul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
y
4atrain/gradients/loss/clip_by_value_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
т
.atrain/gradients/loss/clip_by_value_grad/zerosFill0atrain/gradients/loss/clip_by_value_grad/Shape_24atrain/gradients/loss/clip_by_value_grad/zeros/Const*'
_output_shapes
:         *
T0*

index_type0
й
5atrain/gradients/loss/clip_by_value_grad/GreaterEqualGreaterEqualloss/clip_by_value/Minimumloss/clip_by_value/y*
T0*'
_output_shapes
:         
Ў
>atrain/gradients/loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/loss/clip_by_value_grad/Shape0atrain/gradients/loss/clip_by_value_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Л
/atrain/gradients/loss/clip_by_value_grad/SelectSelect5atrain/gradients/loss/clip_by_value_grad/GreaterEqual7atrain/gradients/loss/mul_grad/tuple/control_dependency.atrain/gradients/loss/clip_by_value_grad/zeros*
T0*'
_output_shapes
:         
Н
1atrain/gradients/loss/clip_by_value_grad/Select_1Select5atrain/gradients/loss/clip_by_value_grad/GreaterEqual.atrain/gradients/loss/clip_by_value_grad/zeros7atrain/gradients/loss/mul_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
ф
,atrain/gradients/loss/clip_by_value_grad/SumSum/atrain/gradients/loss/clip_by_value_grad/Select>atrain/gradients/loss/clip_by_value_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┘
0atrain/gradients/loss/clip_by_value_grad/ReshapeReshape,atrain/gradients/loss/clip_by_value_grad/Sum.atrain/gradients/loss/clip_by_value_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
ъ
.atrain/gradients/loss/clip_by_value_grad/Sum_1Sum1atrain/gradients/loss/clip_by_value_grad/Select_1@atrain/gradients/loss/clip_by_value_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╬
2atrain/gradients/loss/clip_by_value_grad/Reshape_1Reshape.atrain/gradients/loss/clip_by_value_grad/Sum_10atrain/gradients/loss/clip_by_value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
й
9atrain/gradients/loss/clip_by_value_grad/tuple/group_depsNoOp1^atrain/gradients/loss/clip_by_value_grad/Reshape3^atrain/gradients/loss/clip_by_value_grad/Reshape_1
▓
Aatrain/gradients/loss/clip_by_value_grad/tuple/control_dependencyIdentity0atrain/gradients/loss/clip_by_value_grad/Reshape:^atrain/gradients/loss/clip_by_value_grad/tuple/group_deps*C
_class9
75loc:@atrain/gradients/loss/clip_by_value_grad/Reshape*'
_output_shapes
:         *
T0
з
Catrain/gradients/loss/clip_by_value_grad/tuple/control_dependency_1Identity2atrain/gradients/loss/clip_by_value_grad/Reshape_1:^atrain/gradients/loss/clip_by_value_grad/tuple/group_deps*E
_class;
97loc:@atrain/gradients/loss/clip_by_value_grad/Reshape_1*
_output_shapes
: *
T0
М
6atrain/gradients/loss/clip_by_value/Minimum_grad/ShapeShapeloss/surrogate/truediv*
_output_shapes
:*
T0*
out_type0
{
8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
╣
8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_2ShapeAatrain/gradients/loss/clip_by_value_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Б
<atrain/gradients/loss/clip_by_value/Minimum_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
·
6atrain/gradients/loss/clip_by_value/Minimum_grad/zerosFill8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_2<atrain/gradients/loss/clip_by_value/Minimum_grad/zeros/Const*'
_output_shapes
:         *
T0*

index_type0
п
:atrain/gradients/loss/clip_by_value/Minimum_grad/LessEqual	LessEqualloss/surrogate/truedivloss/clip_by_value/Minimum/y*
T0*'
_output_shapes
:         
О
Fatrain/gradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs6atrain/gradients/loss/clip_by_value/Minimum_grad/Shape8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_1*2
_output_shapes 
:         :         *
T0
к
7atrain/gradients/loss/clip_by_value/Minimum_grad/SelectSelect:atrain/gradients/loss/clip_by_value/Minimum_grad/LessEqualAatrain/gradients/loss/clip_by_value_grad/tuple/control_dependency6atrain/gradients/loss/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:         *
T0
м
9atrain/gradients/loss/clip_by_value/Minimum_grad/Select_1Select:atrain/gradients/loss/clip_by_value/Minimum_grad/LessEqual6atrain/gradients/loss/clip_by_value/Minimum_grad/zerosAatrain/gradients/loss/clip_by_value_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
№
4atrain/gradients/loss/clip_by_value/Minimum_grad/SumSum7atrain/gradients/loss/clip_by_value/Minimum_grad/SelectFatrain/gradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ё
8atrain/gradients/loss/clip_by_value/Minimum_grad/ReshapeReshape4atrain/gradients/loss/clip_by_value/Minimum_grad/Sum6atrain/gradients/loss/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
В
6atrain/gradients/loss/clip_by_value/Minimum_grad/Sum_1Sum9atrain/gradients/loss/clip_by_value/Minimum_grad/Select_1Hatrain/gradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ц
:atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1Reshape6atrain/gradients/loss/clip_by_value/Minimum_grad/Sum_18atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
┴
Aatrain/gradients/loss/clip_by_value/Minimum_grad/tuple/group_depsNoOp9^atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape;^atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1
╥
Iatrain/gradients/loss/clip_by_value/Minimum_grad/tuple/control_dependencyIdentity8atrain/gradients/loss/clip_by_value/Minimum_grad/ReshapeB^atrain/gradients/loss/clip_by_value/Minimum_grad/tuple/group_deps*
T0*K
_classA
?=loc:@atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape*'
_output_shapes
:         
╟
Katrain/gradients/loss/clip_by_value/Minimum_grad/tuple/control_dependency_1Identity:atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1B^atrain/gradients/loss/clip_by_value/Minimum_grad/tuple/group_deps*
_output_shapes
: *
T0*M
_classC
A?loc:@atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1
л
atrain/gradients/AddNAddNAatrain/gradients/loss/surrogate/mul_grad/tuple/control_dependencyIatrain/gradients/loss/clip_by_value/Minimum_grad/tuple/control_dependency*
T0*C
_class9
75loc:@atrain/gradients/loss/surrogate/mul_grad/Reshape*
N*'
_output_shapes
:         
Д
2atrain/gradients/loss/surrogate/truediv_grad/ShapeShapepi/Normal/prob/Exp*
_output_shapes
:*
T0*
out_type0
Й
4atrain/gradients/loss/surrogate/truediv_grad/Shape_1Shapeoldpi/Normal/prob/Exp*
_output_shapes
:*
T0*
out_type0
В
Batrain/gradients/loss/surrogate/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs2atrain/gradients/loss/surrogate/truediv_grad/Shape4atrain/gradients/loss/surrogate/truediv_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Я
4atrain/gradients/loss/surrogate/truediv_grad/RealDivRealDivatrain/gradients/AddNoldpi/Normal/prob/Exp*
T0*'
_output_shapes
:         
ё
0atrain/gradients/loss/surrogate/truediv_grad/SumSum4atrain/gradients/loss/surrogate/truediv_grad/RealDivBatrain/gradients/loss/surrogate/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
х
4atrain/gradients/loss/surrogate/truediv_grad/ReshapeReshape0atrain/gradients/loss/surrogate/truediv_grad/Sum2atrain/gradients/loss/surrogate/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
}
0atrain/gradients/loss/surrogate/truediv_grad/NegNegpi/Normal/prob/Exp*'
_output_shapes
:         *
T0
╝
6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_1RealDiv0atrain/gradients/loss/surrogate/truediv_grad/Negoldpi/Normal/prob/Exp*
T0*'
_output_shapes
:         
┬
6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_2RealDiv6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_1oldpi/Normal/prob/Exp*
T0*'
_output_shapes
:         
╕
0atrain/gradients/loss/surrogate/truediv_grad/mulMulatrain/gradients/AddN6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:         
ё
2atrain/gradients/loss/surrogate/truediv_grad/Sum_1Sum0atrain/gradients/loss/surrogate/truediv_grad/mulDatrain/gradients/loss/surrogate/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ы
6atrain/gradients/loss/surrogate/truediv_grad/Reshape_1Reshape2atrain/gradients/loss/surrogate/truediv_grad/Sum_14atrain/gradients/loss/surrogate/truediv_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
╡
=atrain/gradients/loss/surrogate/truediv_grad/tuple/group_depsNoOp5^atrain/gradients/loss/surrogate/truediv_grad/Reshape7^atrain/gradients/loss/surrogate/truediv_grad/Reshape_1
┬
Eatrain/gradients/loss/surrogate/truediv_grad/tuple/control_dependencyIdentity4atrain/gradients/loss/surrogate/truediv_grad/Reshape>^atrain/gradients/loss/surrogate/truediv_grad/tuple/group_deps*
T0*G
_class=
;9loc:@atrain/gradients/loss/surrogate/truediv_grad/Reshape*'
_output_shapes
:         
╚
Gatrain/gradients/loss/surrogate/truediv_grad/tuple/control_dependency_1Identity6atrain/gradients/loss/surrogate/truediv_grad/Reshape_1>^atrain/gradients/loss/surrogate/truediv_grad/tuple/group_deps*'
_output_shapes
:         *
T0*I
_class?
=;loc:@atrain/gradients/loss/surrogate/truediv_grad/Reshape_1
└
,atrain/gradients/pi/Normal/prob/Exp_grad/mulMulEatrain/gradients/loss/surrogate/truediv_grad/tuple/control_dependencypi/Normal/prob/Exp*
T0*'
_output_shapes
:         
А
.atrain/gradients/pi/Normal/prob/sub_grad/ShapeShapepi/Normal/prob/mul*
_output_shapes
:*
T0*
out_type0
В
0atrain/gradients/pi/Normal/prob/sub_grad/Shape_1Shapepi/Normal/prob/add*
_output_shapes
:*
T0*
out_type0
Ў
>atrain/gradients/pi/Normal/prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/pi/Normal/prob/sub_grad/Shape0atrain/gradients/pi/Normal/prob/sub_grad/Shape_1*2
_output_shapes 
:         :         *
T0
с
,atrain/gradients/pi/Normal/prob/sub_grad/SumSum,atrain/gradients/pi/Normal/prob/Exp_grad/mul>atrain/gradients/pi/Normal/prob/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┘
0atrain/gradients/pi/Normal/prob/sub_grad/ReshapeReshape,atrain/gradients/pi/Normal/prob/sub_grad/Sum.atrain/gradients/pi/Normal/prob/sub_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
х
.atrain/gradients/pi/Normal/prob/sub_grad/Sum_1Sum,atrain/gradients/pi/Normal/prob/Exp_grad/mul@atrain/gradients/pi/Normal/prob/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
,atrain/gradients/pi/Normal/prob/sub_grad/NegNeg.atrain/gradients/pi/Normal/prob/sub_grad/Sum_1*
_output_shapes
:*
T0
▌
2atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1Reshape,atrain/gradients/pi/Normal/prob/sub_grad/Neg0atrain/gradients/pi/Normal/prob/sub_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
й
9atrain/gradients/pi/Normal/prob/sub_grad/tuple/group_depsNoOp1^atrain/gradients/pi/Normal/prob/sub_grad/Reshape3^atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1
▓
Aatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependencyIdentity0atrain/gradients/pi/Normal/prob/sub_grad/Reshape:^atrain/gradients/pi/Normal/prob/sub_grad/tuple/group_deps*
T0*C
_class9
75loc:@atrain/gradients/pi/Normal/prob/sub_grad/Reshape*'
_output_shapes
:         
╕
Catrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1:^atrain/gradients/pi/Normal/prob/sub_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1*'
_output_shapes
:         
q
.atrain/gradients/pi/Normal/prob/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Е
0atrain/gradients/pi/Normal/prob/mul_grad/Shape_1Shapepi/Normal/prob/Square*
T0*
out_type0*
_output_shapes
:
Ў
>atrain/gradients/pi/Normal/prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/pi/Normal/prob/mul_grad/Shape0atrain/gradients/pi/Normal/prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┐
,atrain/gradients/pi/Normal/prob/mul_grad/MulMulAatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependencypi/Normal/prob/Square*'
_output_shapes
:         *
T0
с
,atrain/gradients/pi/Normal/prob/mul_grad/SumSum,atrain/gradients/pi/Normal/prob/mul_grad/Mul>atrain/gradients/pi/Normal/prob/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╚
0atrain/gradients/pi/Normal/prob/mul_grad/ReshapeReshape,atrain/gradients/pi/Normal/prob/mul_grad/Sum.atrain/gradients/pi/Normal/prob/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
└
.atrain/gradients/pi/Normal/prob/mul_grad/Mul_1Mulpi/Normal/prob/mul/xAatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
ч
.atrain/gradients/pi/Normal/prob/mul_grad/Sum_1Sum.atrain/gradients/pi/Normal/prob/mul_grad/Mul_1@atrain/gradients/pi/Normal/prob/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
▀
2atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1Reshape.atrain/gradients/pi/Normal/prob/mul_grad/Sum_10atrain/gradients/pi/Normal/prob/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
й
9atrain/gradients/pi/Normal/prob/mul_grad/tuple/group_depsNoOp1^atrain/gradients/pi/Normal/prob/mul_grad/Reshape3^atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1
б
Aatrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependencyIdentity0atrain/gradients/pi/Normal/prob/mul_grad/Reshape:^atrain/gradients/pi/Normal/prob/mul_grad/tuple/group_deps*
T0*C
_class9
75loc:@atrain/gradients/pi/Normal/prob/mul_grad/Reshape*
_output_shapes
: 
╕
Catrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1:^atrain/gradients/pi/Normal/prob/mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1*'
_output_shapes
:         
q
.atrain/gradients/pi/Normal/prob/add_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
В
0atrain/gradients/pi/Normal/prob/add_grad/Shape_1Shapepi/Normal/prob/Log*
T0*
out_type0*
_output_shapes
:
Ў
>atrain/gradients/pi/Normal/prob/add_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/pi/Normal/prob/add_grad/Shape0atrain/gradients/pi/Normal/prob/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
°
,atrain/gradients/pi/Normal/prob/add_grad/SumSumCatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency_1>atrain/gradients/pi/Normal/prob/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╚
0atrain/gradients/pi/Normal/prob/add_grad/ReshapeReshape,atrain/gradients/pi/Normal/prob/add_grad/Sum.atrain/gradients/pi/Normal/prob/add_grad/Shape*
_output_shapes
: *
T0*
Tshape0
№
.atrain/gradients/pi/Normal/prob/add_grad/Sum_1SumCatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency_1@atrain/gradients/pi/Normal/prob/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
▀
2atrain/gradients/pi/Normal/prob/add_grad/Reshape_1Reshape.atrain/gradients/pi/Normal/prob/add_grad/Sum_10atrain/gradients/pi/Normal/prob/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
й
9atrain/gradients/pi/Normal/prob/add_grad/tuple/group_depsNoOp1^atrain/gradients/pi/Normal/prob/add_grad/Reshape3^atrain/gradients/pi/Normal/prob/add_grad/Reshape_1
б
Aatrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependencyIdentity0atrain/gradients/pi/Normal/prob/add_grad/Reshape:^atrain/gradients/pi/Normal/prob/add_grad/tuple/group_deps*
T0*C
_class9
75loc:@atrain/gradients/pi/Normal/prob/add_grad/Reshape*
_output_shapes
: 
╕
Catrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/Normal/prob/add_grad/Reshape_1:^atrain/gradients/pi/Normal/prob/add_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/pi/Normal/prob/add_grad/Reshape_1*'
_output_shapes
:         
╝
1atrain/gradients/pi/Normal/prob/Square_grad/ConstConstD^atrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
┐
/atrain/gradients/pi/Normal/prob/Square_grad/MulMul"pi/Normal/prob/standardize/truediv1atrain/gradients/pi/Normal/prob/Square_grad/Const*
T0*'
_output_shapes
:         
р
1atrain/gradients/pi/Normal/prob/Square_grad/Mul_1MulCatrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependency_1/atrain/gradients/pi/Normal/prob/Square_grad/Mul*'
_output_shapes
:         *
T0
╩
3atrain/gradients/pi/Normal/prob/Log_grad/Reciprocal
Reciprocalpi/Normal/scaleD^atrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
▀
,atrain/gradients/pi/Normal/prob/Log_grad/mulMulCatrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependency_13atrain/gradients/pi/Normal/prob/Log_grad/Reciprocal*'
_output_shapes
:         *
T0
Ь
>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ShapeShapepi/Normal/prob/standardize/sub*
T0*
out_type0*
_output_shapes
:
П
@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape_1Shapepi/Normal/scale*
T0*
out_type0*
_output_shapes
:
ж
Natrain/gradients/pi/Normal/prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┴
@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDivRealDiv1atrain/gradients/pi/Normal/prob/Square_grad/Mul_1pi/Normal/scale*
T0*'
_output_shapes
:         
Х
<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/SumSum@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDivNatrain/gradients/pi/Normal/prob/standardize/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Й
@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ReshapeReshape<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Sum>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Х
<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/NegNegpi/Normal/prob/standardize/sub*'
_output_shapes
:         *
T0
╬
Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_1RealDiv<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Negpi/Normal/scale*
T0*'
_output_shapes
:         
╘
Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_2RealDivBatrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_1pi/Normal/scale*
T0*'
_output_shapes
:         
ь
<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/mulMul1atrain/gradients/pi/Normal/prob/Square_grad/Mul_1Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_2*'
_output_shapes
:         *
T0
Х
>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Sum_1Sum<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/mulPatrain/gradients/pi/Normal/prob/standardize/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
П
Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1Reshape>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Sum_1@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
┘
Iatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/group_depsNoOpA^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ReshapeC^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1
Є
Qatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependencyIdentity@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ReshapeJ^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/group_deps*
T0*S
_classI
GEloc:@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape*'
_output_shapes
:         
°
Satrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependency_1IdentityBatrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1J^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/group_deps*U
_classK
IGloc:@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1*'
_output_shapes
:         *
T0
А
:atrain/gradients/pi/Normal/prob/standardize/sub_grad/ShapeShapeaction*
_output_shapes
:*
T0*
out_type0
Й
<atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape_1Shapepi/Normal/loc*
T0*
out_type0*
_output_shapes
:
Ъ
Jatrain/gradients/pi/Normal/prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape<atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ю
8atrain/gradients/pi/Normal/prob/standardize/sub_grad/SumSumQatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependencyJatrain/gradients/pi/Normal/prob/standardize/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
¤
<atrain/gradients/pi/Normal/prob/standardize/sub_grad/ReshapeReshape8atrain/gradients/pi/Normal/prob/standardize/sub_grad/Sum:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
в
:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Sum_1SumQatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependencyLatrain/gradients/pi/Normal/prob/standardize/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ю
8atrain/gradients/pi/Normal/prob/standardize/sub_grad/NegNeg:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:
Б
>atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1Reshape8atrain/gradients/pi/Normal/prob/standardize/sub_grad/Neg<atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
═
Eatrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/group_depsNoOp=^atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape?^atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1
т
Matrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/control_dependencyIdentity<atrain/gradients/pi/Normal/prob/standardize/sub_grad/ReshapeF^atrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/group_deps*
T0*O
_classE
CAloc:@atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape*'
_output_shapes
:         
ш
Oatrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/control_dependency_1Identity>atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1F^atrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/group_deps*'
_output_shapes
:         *
T0*Q
_classG
ECloc:@atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1
Ю
atrain/gradients/AddN_1AddN,atrain/gradients/pi/Normal/prob/Log_grad/mulSatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependency_1*?
_class5
31loc:@atrain/gradients/pi/Normal/prob/Log_grad/mul*
N*'
_output_shapes
:         *
T0
б
4atrain/gradients/pi/dense/Softplus_grad/SoftplusGradSoftplusGradatrain/gradients/AddN_1pi/dense/BiasAdd*
T0*'
_output_shapes
:         
┬
(atrain/gradients/pi/a/Tanh_grad/TanhGradTanhGrad	pi/a/TanhOatrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
│
2atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad4atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad*
T0*
data_formatNHWC*
_output_shapes
:
л
7atrain/gradients/pi/dense/BiasAdd_grad/tuple/group_depsNoOp3^atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad
╢
?atrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity4atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad8^atrain/gradients/pi/dense/BiasAdd_grad/tuple/group_deps*G
_class=
;9loc:@atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad*'
_output_shapes
:         *
T0
з
Aatrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGrad8^atrain/gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
г
.atrain/gradients/pi/a/BiasAdd_grad/BiasAddGradBiasAddGrad(atrain/gradients/pi/a/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:
Ч
3atrain/gradients/pi/a/BiasAdd_grad/tuple/group_depsNoOp/^atrain/gradients/pi/a/BiasAdd_grad/BiasAddGrad)^atrain/gradients/pi/a/Tanh_grad/TanhGrad
Ц
;atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependencyIdentity(atrain/gradients/pi/a/Tanh_grad/TanhGrad4^atrain/gradients/pi/a/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@atrain/gradients/pi/a/Tanh_grad/TanhGrad*'
_output_shapes
:         
Ч
=atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependency_1Identity.atrain/gradients/pi/a/BiasAdd_grad/BiasAddGrad4^atrain/gradients/pi/a/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@atrain/gradients/pi/a/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ц
,atrain/gradients/pi/dense/MatMul_grad/MatMulMatMul?atrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependencypi/dense/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b(*
T0
╒
.atrain/gradients/pi/dense/MatMul_grad/MatMul_1MatMul
pi/l4/Relu?atrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	А*
transpose_a(*
transpose_b( 
Ю
6atrain/gradients/pi/dense/MatMul_grad/tuple/group_depsNoOp-^atrain/gradients/pi/dense/MatMul_grad/MatMul/^atrain/gradients/pi/dense/MatMul_grad/MatMul_1
е
>atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependencyIdentity,atrain/gradients/pi/dense/MatMul_grad/MatMul7^atrain/gradients/pi/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@atrain/gradients/pi/dense/MatMul_grad/MatMul*(
_output_shapes
:         А
в
@atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Identity.atrain/gradients/pi/dense/MatMul_grad/MatMul_17^atrain/gradients/pi/dense/MatMul_grad/tuple/group_deps*A
_class7
53loc:@atrain/gradients/pi/dense/MatMul_grad/MatMul_1*
_output_shapes
:	А*
T0
┌
(atrain/gradients/pi/a/MatMul_grad/MatMulMatMul;atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependencypi/a/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b(*
T0
═
*atrain/gradients/pi/a/MatMul_grad/MatMul_1MatMul
pi/l4/Relu;atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	А*
transpose_a(*
transpose_b( *
T0
Т
2atrain/gradients/pi/a/MatMul_grad/tuple/group_depsNoOp)^atrain/gradients/pi/a/MatMul_grad/MatMul+^atrain/gradients/pi/a/MatMul_grad/MatMul_1
Х
:atrain/gradients/pi/a/MatMul_grad/tuple/control_dependencyIdentity(atrain/gradients/pi/a/MatMul_grad/MatMul3^atrain/gradients/pi/a/MatMul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*;
_class1
/-loc:@atrain/gradients/pi/a/MatMul_grad/MatMul
Т
<atrain/gradients/pi/a/MatMul_grad/tuple/control_dependency_1Identity*atrain/gradients/pi/a/MatMul_grad/MatMul_13^atrain/gradients/pi/a/MatMul_grad/tuple/group_deps*
_output_shapes
:	А*
T0*=
_class3
1/loc:@atrain/gradients/pi/a/MatMul_grad/MatMul_1
Ш
atrain/gradients/AddN_2AddN>atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependency:atrain/gradients/pi/a/MatMul_grad/tuple/control_dependency*
N*(
_output_shapes
:         А*
T0*?
_class5
31loc:@atrain/gradients/pi/dense/MatMul_grad/MatMul
Н
)atrain/gradients/pi/l4/Relu_grad/ReluGradReluGradatrain/gradients/AddN_2
pi/l4/Relu*
T0*(
_output_shapes
:         А
ж
/atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l4/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:А*
T0
Ъ
4atrain/gradients/pi/l4/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l4/Relu_grad/ReluGrad
Ы
<atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l4/Relu_grad/ReluGrad5^atrain/gradients/pi/l4/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*<
_class2
0.loc:@atrain/gradients/pi/l4/Relu_grad/ReluGrad
Ь
>atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l4/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А*
T0
▌
)atrain/gradients/pi/l4/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependencypi/l4/kernel/read*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b(
╨
+atrain/gradients/pi/l4/MatMul_grad/MatMul_1MatMul
pi/l3/Relu<atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( 
Х
3atrain/gradients/pi/l4/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l4/MatMul_grad/MatMul,^atrain/gradients/pi/l4/MatMul_grad/MatMul_1
Щ
;atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l4/MatMul_grad/MatMul4^atrain/gradients/pi/l4/MatMul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*<
_class2
0.loc:@atrain/gradients/pi/l4/MatMul_grad/MatMul
Ч
=atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l4/MatMul_grad/MatMul_14^atrain/gradients/pi/l4/MatMul_grad/tuple/group_deps* 
_output_shapes
:
АА*
T0*>
_class4
20loc:@atrain/gradients/pi/l4/MatMul_grad/MatMul_1
▒
)atrain/gradients/pi/l3/Relu_grad/ReluGradReluGrad;atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependency
pi/l3/Relu*(
_output_shapes
:         А*
T0
ж
/atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l3/Relu_grad/ReluGrad*
_output_shapes	
:А*
T0*
data_formatNHWC
Ъ
4atrain/gradients/pi/l3/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l3/Relu_grad/ReluGrad
Ы
<atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l3/Relu_grad/ReluGrad5^atrain/gradients/pi/l3/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@atrain/gradients/pi/l3/Relu_grad/ReluGrad*(
_output_shapes
:         А*
T0
Ь
>atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l3/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
▌
)atrain/gradients/pi/l3/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependencypi/l3/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b(*
T0
╨
+atrain/gradients/pi/l3/MatMul_grad/MatMul_1MatMul
pi/l2/Relu<atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( *
T0
Х
3atrain/gradients/pi/l3/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l3/MatMul_grad/MatMul,^atrain/gradients/pi/l3/MatMul_grad/MatMul_1
Щ
;atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l3/MatMul_grad/MatMul4^atrain/gradients/pi/l3/MatMul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*<
_class2
0.loc:@atrain/gradients/pi/l3/MatMul_grad/MatMul
Ч
=atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l3/MatMul_grad/MatMul_14^atrain/gradients/pi/l3/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@atrain/gradients/pi/l3/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
▒
)atrain/gradients/pi/l2/Relu_grad/ReluGradReluGrad;atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependency
pi/l2/Relu*
T0*(
_output_shapes
:         А
ж
/atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:А*
T0
Ъ
4atrain/gradients/pi/l2/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l2/Relu_grad/ReluGrad
Ы
<atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l2/Relu_grad/ReluGrad5^atrain/gradients/pi/l2/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*<
_class2
0.loc:@atrain/gradients/pi/l2/Relu_grad/ReluGrad
Ь
>atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l2/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*B
_class8
64loc:@atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGrad
▌
)atrain/gradients/pi/l2/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependencypi/l2/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b(*
T0
╨
+atrain/gradients/pi/l2/MatMul_grad/MatMul_1MatMul
pi/l1/Relu<atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( *
T0
Х
3atrain/gradients/pi/l2/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l2/MatMul_grad/MatMul,^atrain/gradients/pi/l2/MatMul_grad/MatMul_1
Щ
;atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l2/MatMul_grad/MatMul4^atrain/gradients/pi/l2/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@atrain/gradients/pi/l2/MatMul_grad/MatMul*(
_output_shapes
:         А*
T0
Ч
=atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l2/MatMul_grad/MatMul_14^atrain/gradients/pi/l2/MatMul_grad/tuple/group_deps*>
_class4
20loc:@atrain/gradients/pi/l2/MatMul_grad/MatMul_1* 
_output_shapes
:
АА*
T0
▒
)atrain/gradients/pi/l1/Relu_grad/ReluGradReluGrad;atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependency
pi/l1/Relu*
T0*(
_output_shapes
:         А
ж
/atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:А*
T0
Ъ
4atrain/gradients/pi/l1/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l1/Relu_grad/ReluGrad
Ы
<atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l1/Relu_grad/ReluGrad5^atrain/gradients/pi/l1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l1/Relu_grad/ReluGrad*(
_output_shapes
:         А
Ь
>atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*B
_class8
64loc:@atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGrad
▄
)atrain/gradients/pi/l1/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependencypi/l1/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b(*
T0
╩
+atrain/gradients/pi/l1/MatMul_grad/MatMul_1MatMulstate<atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	А*
transpose_a(*
transpose_b( *
T0
Х
3atrain/gradients/pi/l1/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l1/MatMul_grad/MatMul,^atrain/gradients/pi/l1/MatMul_grad/MatMul_1
Ш
;atrain/gradients/pi/l1/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l1/MatMul_grad/MatMul4^atrain/gradients/pi/l1/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l1/MatMul_grad/MatMul*'
_output_shapes
:         
Ц
=atrain/gradients/pi/l1/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l1/MatMul_grad/MatMul_14^atrain/gradients/pi/l1/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@atrain/gradients/pi/l1/MatMul_grad/MatMul_1*
_output_shapes
:	А
Г
 atrain/beta1_power/initial_valueConst*
_class
loc:@pi/a/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Ф
atrain/beta1_power
VariableV2*
shared_name *
_class
loc:@pi/a/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
┴
atrain/beta1_power/AssignAssignatrain/beta1_power atrain/beta1_power/initial_value*
use_locking(*
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
: 
v
atrain/beta1_power/readIdentityatrain/beta1_power*
_output_shapes
: *
T0*
_class
loc:@pi/a/bias
Г
 atrain/beta2_power/initial_valueConst*
_class
loc:@pi/a/bias*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
Ф
atrain/beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@pi/a/bias*
	container 
┴
atrain/beta2_power/AssignAssignatrain/beta2_power atrain/beta2_power/initial_value*
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
: *
use_locking(
v
atrain/beta2_power/readIdentityatrain/beta2_power*
T0*
_class
loc:@pi/a/bias*
_output_shapes
: 
м
:atrain/pi/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ц
0atrain/pi/l1/kernel/Adam/Initializer/zeros/ConstConst*
_class
loc:@pi/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
*atrain/pi/l1/kernel/Adam/Initializer/zerosFill:atrain/pi/l1/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l1/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	А*
T0*
_class
loc:@pi/l1/kernel*

index_type0
п
atrain/pi/l1/kernel/Adam
VariableV2*
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *
_class
loc:@pi/l1/kernel*
	container 
у
atrain/pi/l1/kernel/Adam/AssignAssignatrain/pi/l1/kernel/Adam*atrain/pi/l1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l1/kernel*
validate_shape(*
_output_shapes
:	А
О
atrain/pi/l1/kernel/Adam/readIdentityatrain/pi/l1/kernel/Adam*
T0*
_class
loc:@pi/l1/kernel*
_output_shapes
:	А
о
<atrain/pi/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ш
2atrain/pi/l1/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@pi/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Г
,atrain/pi/l1/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l1/kernel/Adam_1/Initializer/zeros/Const*
_class
loc:@pi/l1/kernel*

index_type0*
_output_shapes
:	А*
T0
▒
atrain/pi/l1/kernel/Adam_1
VariableV2*
shared_name *
_class
loc:@pi/l1/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
щ
!atrain/pi/l1/kernel/Adam_1/AssignAssignatrain/pi/l1/kernel/Adam_1,atrain/pi/l1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l1/kernel*
validate_shape(*
_output_shapes
:	А
Т
atrain/pi/l1/kernel/Adam_1/readIdentityatrain/pi/l1/kernel/Adam_1*
T0*
_class
loc:@pi/l1/kernel*
_output_shapes
:	А
Ц
(atrain/pi/l1/bias/Adam/Initializer/zerosConst*
_class
loc:@pi/l1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
г
atrain/pi/l1/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l1/bias*
	container *
shape:А
╫
atrain/pi/l1/bias/Adam/AssignAssignatrain/pi/l1/bias/Adam(atrain/pi/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l1/bias*
validate_shape(*
_output_shapes	
:А
Д
atrain/pi/l1/bias/Adam/readIdentityatrain/pi/l1/bias/Adam*
_output_shapes	
:А*
T0*
_class
loc:@pi/l1/bias
Ш
*atrain/pi/l1/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:А*
_class
loc:@pi/l1/bias*
valueBА*    *
dtype0
е
atrain/pi/l1/bias/Adam_1
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l1/bias*
	container 
▌
atrain/pi/l1/bias/Adam_1/AssignAssignatrain/pi/l1/bias/Adam_1*atrain/pi/l1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l1/bias*
validate_shape(*
_output_shapes	
:А
И
atrain/pi/l1/bias/Adam_1/readIdentityatrain/pi/l1/bias/Adam_1*
T0*
_class
loc:@pi/l1/bias*
_output_shapes	
:А
м
:atrain/pi/l2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ц
0atrain/pi/l2/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
_class
loc:@pi/l2/kernel*
valueB
 *    *
dtype0
■
*atrain/pi/l2/kernel/Adam/Initializer/zerosFill:atrain/pi/l2/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l2/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
АА*
T0*
_class
loc:@pi/l2/kernel*

index_type0
▒
atrain/pi/l2/kernel/Adam
VariableV2*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *
_class
loc:@pi/l2/kernel
ф
atrain/pi/l2/kernel/Adam/AssignAssignatrain/pi/l2/kernel/Adam*atrain/pi/l2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l2/kernel*
validate_shape(* 
_output_shapes
:
АА
П
atrain/pi/l2/kernel/Adam/readIdentityatrain/pi/l2/kernel/Adam*
T0*
_class
loc:@pi/l2/kernel* 
_output_shapes
:
АА
о
<atrain/pi/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ш
2atrain/pi/l2/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
_class
loc:@pi/l2/kernel*
valueB
 *    *
dtype0
Д
,atrain/pi/l2/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l2/kernel/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@pi/l2/kernel*

index_type0* 
_output_shapes
:
АА
│
atrain/pi/l2/kernel/Adam_1
VariableV2*
shared_name *
_class
loc:@pi/l2/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
ъ
!atrain/pi/l2/kernel/Adam_1/AssignAssignatrain/pi/l2/kernel/Adam_1,atrain/pi/l2/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*
_class
loc:@pi/l2/kernel
У
atrain/pi/l2/kernel/Adam_1/readIdentityatrain/pi/l2/kernel/Adam_1*
T0*
_class
loc:@pi/l2/kernel* 
_output_shapes
:
АА
Ц
(atrain/pi/l2/bias/Adam/Initializer/zerosConst*
_class
loc:@pi/l2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
г
atrain/pi/l2/bias/Adam
VariableV2*
shared_name *
_class
loc:@pi/l2/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
╫
atrain/pi/l2/bias/Adam/AssignAssignatrain/pi/l2/bias/Adam(atrain/pi/l2/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l2/bias*
validate_shape(*
_output_shapes	
:А
Д
atrain/pi/l2/bias/Adam/readIdentityatrain/pi/l2/bias/Adam*
_class
loc:@pi/l2/bias*
_output_shapes	
:А*
T0
Ш
*atrain/pi/l2/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
е
atrain/pi/l2/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@pi/l2/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
▌
atrain/pi/l2/bias/Adam_1/AssignAssignatrain/pi/l2/bias/Adam_1*atrain/pi/l2/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@pi/l2/bias
И
atrain/pi/l2/bias/Adam_1/readIdentityatrain/pi/l2/bias/Adam_1*
_class
loc:@pi/l2/bias*
_output_shapes	
:А*
T0
м
:atrain/pi/l3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
_class
loc:@pi/l3/kernel*
valueB"      *
dtype0
Ц
0atrain/pi/l3/kernel/Adam/Initializer/zeros/ConstConst*
_class
loc:@pi/l3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
■
*atrain/pi/l3/kernel/Adam/Initializer/zerosFill:atrain/pi/l3/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l3/kernel/Adam/Initializer/zeros/Const*
_class
loc:@pi/l3/kernel*

index_type0* 
_output_shapes
:
АА*
T0
▒
atrain/pi/l3/kernel/Adam
VariableV2*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *
_class
loc:@pi/l3/kernel
ф
atrain/pi/l3/kernel/Adam/AssignAssignatrain/pi/l3/kernel/Adam*atrain/pi/l3/kernel/Adam/Initializer/zeros*
_class
loc:@pi/l3/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0
П
atrain/pi/l3/kernel/Adam/readIdentityatrain/pi/l3/kernel/Adam* 
_output_shapes
:
АА*
T0*
_class
loc:@pi/l3/kernel
о
<atrain/pi/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ш
2atrain/pi/l3/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@pi/l3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Д
,atrain/pi/l3/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l3/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
АА*
T0*
_class
loc:@pi/l3/kernel*

index_type0
│
atrain/pi/l3/kernel/Adam_1
VariableV2*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *
_class
loc:@pi/l3/kernel
ъ
!atrain/pi/l3/kernel/Adam_1/AssignAssignatrain/pi/l3/kernel/Adam_1,atrain/pi/l3/kernel/Adam_1/Initializer/zeros*
_class
loc:@pi/l3/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0
У
atrain/pi/l3/kernel/Adam_1/readIdentityatrain/pi/l3/kernel/Adam_1*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:
АА
Ц
(atrain/pi/l3/bias/Adam/Initializer/zerosConst*
_class
loc:@pi/l3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
г
atrain/pi/l3/bias/Adam
VariableV2*
_class
loc:@pi/l3/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
╫
atrain/pi/l3/bias/Adam/AssignAssignatrain/pi/l3/bias/Adam(atrain/pi/l3/bias/Adam/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@pi/l3/bias*
validate_shape(
Д
atrain/pi/l3/bias/Adam/readIdentityatrain/pi/l3/bias/Adam*
T0*
_class
loc:@pi/l3/bias*
_output_shapes	
:А
Ш
*atrain/pi/l3/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
е
atrain/pi/l3/bias/Adam_1
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l3/bias*
	container 
▌
atrain/pi/l3/bias/Adam_1/AssignAssignatrain/pi/l3/bias/Adam_1*atrain/pi/l3/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l3/bias*
validate_shape(*
_output_shapes	
:А
И
atrain/pi/l3/bias/Adam_1/readIdentityatrain/pi/l3/bias/Adam_1*
_output_shapes	
:А*
T0*
_class
loc:@pi/l3/bias
м
:atrain/pi/l4/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l4/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
Ц
0atrain/pi/l4/kernel/Adam/Initializer/zeros/ConstConst*
_class
loc:@pi/l4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
■
*atrain/pi/l4/kernel/Adam/Initializer/zerosFill:atrain/pi/l4/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l4/kernel/Adam/Initializer/zeros/Const*
T0*
_class
loc:@pi/l4/kernel*

index_type0* 
_output_shapes
:
АА
▒
atrain/pi/l4/kernel/Adam
VariableV2*
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *
_class
loc:@pi/l4/kernel*
	container 
ф
atrain/pi/l4/kernel/Adam/AssignAssignatrain/pi/l4/kernel/Adam*atrain/pi/l4/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*
_class
loc:@pi/l4/kernel
П
atrain/pi/l4/kernel/Adam/readIdentityatrain/pi/l4/kernel/Adam*
T0*
_class
loc:@pi/l4/kernel* 
_output_shapes
:
АА
о
<atrain/pi/l4/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@pi/l4/kernel*
valueB"   А   
Ш
2atrain/pi/l4/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@pi/l4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Д
,atrain/pi/l4/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l4/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l4/kernel/Adam_1/Initializer/zeros/Const*
_class
loc:@pi/l4/kernel*

index_type0* 
_output_shapes
:
АА*
T0
│
atrain/pi/l4/kernel/Adam_1
VariableV2*
_class
loc:@pi/l4/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name 
ъ
!atrain/pi/l4/kernel/Adam_1/AssignAssignatrain/pi/l4/kernel/Adam_1,atrain/pi/l4/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l4/kernel*
validate_shape(* 
_output_shapes
:
АА
У
atrain/pi/l4/kernel/Adam_1/readIdentityatrain/pi/l4/kernel/Adam_1*
T0*
_class
loc:@pi/l4/kernel* 
_output_shapes
:
АА
Ц
(atrain/pi/l4/bias/Adam/Initializer/zerosConst*
_class
loc:@pi/l4/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
г
atrain/pi/l4/bias/Adam
VariableV2*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l4/bias*
	container *
shape:А*
dtype0
╫
atrain/pi/l4/bias/Adam/AssignAssignatrain/pi/l4/bias/Adam(atrain/pi/l4/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l4/bias*
validate_shape(*
_output_shapes	
:А
Д
atrain/pi/l4/bias/Adam/readIdentityatrain/pi/l4/bias/Adam*
_output_shapes	
:А*
T0*
_class
loc:@pi/l4/bias
Ш
*atrain/pi/l4/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l4/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
е
atrain/pi/l4/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@pi/l4/bias*
	container *
shape:А
▌
atrain/pi/l4/bias/Adam_1/AssignAssignatrain/pi/l4/bias/Adam_1*atrain/pi/l4/bias/Adam_1/Initializer/zeros*
T0*
_class
loc:@pi/l4/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
И
atrain/pi/l4/bias/Adam_1/readIdentityatrain/pi/l4/bias/Adam_1*
_output_shapes	
:А*
T0*
_class
loc:@pi/l4/bias
а
)atrain/pi/a/kernel/Adam/Initializer/zerosConst*
_class
loc:@pi/a/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
н
atrain/pi/a/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *
_class
loc:@pi/a/kernel*
	container *
shape:	А
▀
atrain/pi/a/kernel/Adam/AssignAssignatrain/pi/a/kernel/Adam)atrain/pi/a/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/a/kernel*
validate_shape(*
_output_shapes
:	А
Л
atrain/pi/a/kernel/Adam/readIdentityatrain/pi/a/kernel/Adam*
_class
loc:@pi/a/kernel*
_output_shapes
:	А*
T0
в
+atrain/pi/a/kernel/Adam_1/Initializer/zerosConst*
_output_shapes
:	А*
_class
loc:@pi/a/kernel*
valueB	А*    *
dtype0
п
atrain/pi/a/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *
_class
loc:@pi/a/kernel*
	container *
shape:	А
х
 atrain/pi/a/kernel/Adam_1/AssignAssignatrain/pi/a/kernel/Adam_1+atrain/pi/a/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/a/kernel*
validate_shape(*
_output_shapes
:	А
П
atrain/pi/a/kernel/Adam_1/readIdentityatrain/pi/a/kernel/Adam_1*
_output_shapes
:	А*
T0*
_class
loc:@pi/a/kernel
Т
'atrain/pi/a/bias/Adam/Initializer/zerosConst*
_class
loc:@pi/a/bias*
valueB*    *
dtype0*
_output_shapes
:
Я
atrain/pi/a/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@pi/a/bias*
	container *
shape:
╥
atrain/pi/a/bias/Adam/AssignAssignatrain/pi/a/bias/Adam'atrain/pi/a/bias/Adam/Initializer/zeros*
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(
А
atrain/pi/a/bias/Adam/readIdentityatrain/pi/a/bias/Adam*
T0*
_class
loc:@pi/a/bias*
_output_shapes
:
Ф
)atrain/pi/a/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/a/bias*
valueB*    *
dtype0*
_output_shapes
:
б
atrain/pi/a/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@pi/a/bias*
	container *
shape:
╪
atrain/pi/a/bias/Adam_1/AssignAssignatrain/pi/a/bias/Adam_1)atrain/pi/a/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
:
Д
atrain/pi/a/bias/Adam_1/readIdentityatrain/pi/a/bias/Adam_1*
T0*
_class
loc:@pi/a/bias*
_output_shapes
:
и
-atrain/pi/dense/kernel/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
╡
atrain/pi/dense/kernel/Adam
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *"
_class
loc:@pi/dense/kernel
я
"atrain/pi/dense/kernel/Adam/AssignAssignatrain/pi/dense/kernel/Adam-atrain/pi/dense/kernel/Adam/Initializer/zeros*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(
Ч
 atrain/pi/dense/kernel/Adam/readIdentityatrain/pi/dense/kernel/Adam*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	А
к
/atrain/pi/dense/kernel/Adam_1/Initializer/zerosConst*"
_class
loc:@pi/dense/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
╖
atrain/pi/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	А
ї
$atrain/pi/dense/kernel/Adam_1/AssignAssignatrain/pi/dense/kernel/Adam_1/atrain/pi/dense/kernel/Adam_1/Initializer/zeros*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0
Ы
"atrain/pi/dense/kernel/Adam_1/readIdentityatrain/pi/dense/kernel/Adam_1*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	А
Ъ
+atrain/pi/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
з
atrain/pi/dense/bias/Adam
VariableV2*
_output_shapes
:*
shared_name * 
_class
loc:@pi/dense/bias*
	container *
shape:*
dtype0
т
 atrain/pi/dense/bias/Adam/AssignAssignatrain/pi/dense/bias/Adam+atrain/pi/dense/bias/Adam/Initializer/zeros*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
М
atrain/pi/dense/bias/Adam/readIdentityatrain/pi/dense/bias/Adam*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:
Ь
-atrain/pi/dense/bias/Adam_1/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
й
atrain/pi/dense/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@pi/dense/bias*
	container *
shape:
ш
"atrain/pi/dense/bias/Adam_1/AssignAssignatrain/pi/dense/bias/Adam_1-atrain/pi/dense/bias/Adam_1/Initializer/zeros* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Р
 atrain/pi/dense/bias/Adam_1/readIdentityatrain/pi/dense/bias/Adam_1*
_output_shapes
:*
T0* 
_class
loc:@pi/dense/bias
^
atrain/Adam/learning_rateConst*
valueB
 *╖╤8*
dtype0*
_output_shapes
: 
V
atrain/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
V
atrain/Adam/beta2Const*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
X
atrain/Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
│
)atrain/Adam/update_pi/l1/kernel/ApplyAdam	ApplyAdampi/l1/kernelatrain/pi/l1/kernel/Adamatrain/pi/l1/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	А*
use_locking( *
T0*
_class
loc:@pi/l1/kernel
ж
'atrain/Adam/update_pi/l1/bias/ApplyAdam	ApplyAdam
pi/l1/biasatrain/pi/l1/bias/Adamatrain/pi/l1/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@pi/l1/bias*
use_nesterov( *
_output_shapes	
:А*
use_locking( 
┤
)atrain/Adam/update_pi/l2/kernel/ApplyAdam	ApplyAdampi/l2/kernelatrain/pi/l2/kernel/Adamatrain/pi/l2/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
АА*
use_locking( *
T0*
_class
loc:@pi/l2/kernel*
use_nesterov( 
ж
'atrain/Adam/update_pi/l2/bias/ApplyAdam	ApplyAdam
pi/l2/biasatrain/pi/l2/bias/Adamatrain/pi/l2/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:А*
use_locking( *
T0*
_class
loc:@pi/l2/bias
┤
)atrain/Adam/update_pi/l3/kernel/ApplyAdam	ApplyAdampi/l3/kernelatrain/pi/l3/kernel/Adamatrain/pi/l3/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependency_1*
_class
loc:@pi/l3/kernel*
use_nesterov( * 
_output_shapes
:
АА*
use_locking( *
T0
ж
'atrain/Adam/update_pi/l3/bias/ApplyAdam	ApplyAdam
pi/l3/biasatrain/pi/l3/bias/Adamatrain/pi/l3/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l3/bias*
use_nesterov( *
_output_shapes	
:А
┤
)atrain/Adam/update_pi/l4/kernel/ApplyAdam	ApplyAdampi/l4/kernelatrain/pi/l4/kernel/Adamatrain/pi/l4/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l4/kernel*
use_nesterov( * 
_output_shapes
:
АА
ж
'atrain/Adam/update_pi/l4/bias/ApplyAdam	ApplyAdam
pi/l4/biasatrain/pi/l4/bias/Adamatrain/pi/l4/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l4/bias*
use_nesterov( *
_output_shapes	
:А
н
(atrain/Adam/update_pi/a/kernel/ApplyAdam	ApplyAdampi/a/kernelatrain/pi/a/kernel/Adamatrain/pi/a/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon<atrain/gradients/pi/a/MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@pi/a/kernel*
use_nesterov( *
_output_shapes
:	А*
use_locking( 
Я
&atrain/Adam/update_pi/a/bias/ApplyAdam	ApplyAdam	pi/a/biasatrain/pi/a/bias/Adamatrain/pi/a/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/a/bias*
use_nesterov( *
_output_shapes
:
┼
,atrain/Adam/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelatrain/pi/dense/kernel/Adamatrain/pi/dense/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon@atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	А*
use_locking( *
T0*"
_class
loc:@pi/dense/kernel
╖
*atrain/Adam/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biasatrain/pi/dense/bias/Adamatrain/pi/dense/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilonAatrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1*
T0* 
_class
loc:@pi/dense/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 
Й
atrain/Adam/mulMulatrain/beta1_power/readatrain/Adam/beta1'^atrain/Adam/update_pi/a/bias/ApplyAdam)^atrain/Adam/update_pi/a/kernel/ApplyAdam+^atrain/Adam/update_pi/dense/bias/ApplyAdam-^atrain/Adam/update_pi/dense/kernel/ApplyAdam(^atrain/Adam/update_pi/l1/bias/ApplyAdam*^atrain/Adam/update_pi/l1/kernel/ApplyAdam(^atrain/Adam/update_pi/l2/bias/ApplyAdam*^atrain/Adam/update_pi/l2/kernel/ApplyAdam(^atrain/Adam/update_pi/l3/bias/ApplyAdam*^atrain/Adam/update_pi/l3/kernel/ApplyAdam(^atrain/Adam/update_pi/l4/bias/ApplyAdam*^atrain/Adam/update_pi/l4/kernel/ApplyAdam*
_class
loc:@pi/a/bias*
_output_shapes
: *
T0
й
atrain/Adam/AssignAssignatrain/beta1_poweratrain/Adam/mul*
use_locking( *
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
: 
Л
atrain/Adam/mul_1Mulatrain/beta2_power/readatrain/Adam/beta2'^atrain/Adam/update_pi/a/bias/ApplyAdam)^atrain/Adam/update_pi/a/kernel/ApplyAdam+^atrain/Adam/update_pi/dense/bias/ApplyAdam-^atrain/Adam/update_pi/dense/kernel/ApplyAdam(^atrain/Adam/update_pi/l1/bias/ApplyAdam*^atrain/Adam/update_pi/l1/kernel/ApplyAdam(^atrain/Adam/update_pi/l2/bias/ApplyAdam*^atrain/Adam/update_pi/l2/kernel/ApplyAdam(^atrain/Adam/update_pi/l3/bias/ApplyAdam*^atrain/Adam/update_pi/l3/kernel/ApplyAdam(^atrain/Adam/update_pi/l4/bias/ApplyAdam*^atrain/Adam/update_pi/l4/kernel/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@pi/a/bias
н
atrain/Adam/Assign_1Assignatrain/beta2_poweratrain/Adam/mul_1*
use_locking( *
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
: 
╟
atrain/AdamNoOp^atrain/Adam/Assign^atrain/Adam/Assign_1'^atrain/Adam/update_pi/a/bias/ApplyAdam)^atrain/Adam/update_pi/a/kernel/ApplyAdam+^atrain/Adam/update_pi/dense/bias/ApplyAdam-^atrain/Adam/update_pi/dense/kernel/ApplyAdam(^atrain/Adam/update_pi/l1/bias/ApplyAdam*^atrain/Adam/update_pi/l1/kernel/ApplyAdam(^atrain/Adam/update_pi/l2/bias/ApplyAdam*^atrain/Adam/update_pi/l2/kernel/ApplyAdam(^atrain/Adam/update_pi/l3/bias/ApplyAdam*^atrain/Adam/update_pi/l3/kernel/ApplyAdam(^atrain/Adam/update_pi/l4/bias/ApplyAdam*^atrain/Adam/update_pi/l4/kernel/ApplyAdam""(
train_op

critic/Adam
atrain/Adam"╙L
	variables┼L┬L
c
critic/w1_s:0critic/w1_s/Assigncritic/w1_s/read:02(critic/w1_s/Initializer/random_uniform:08
[
critic/b1:0critic/b1/Assigncritic/b1/read:02&critic/b1/Initializer/random_uniform:08
w
critic/l2/kernel:0critic/l2/kernel/Assigncritic/l2/kernel/read:02-critic/l2/kernel/Initializer/random_uniform:08
f
critic/l2/bias:0critic/l2/bias/Assigncritic/l2/bias/read:02"critic/l2/bias/Initializer/zeros:08
w
critic/l3/kernel:0critic/l3/kernel/Assigncritic/l3/kernel/read:02-critic/l3/kernel/Initializer/random_uniform:08
f
critic/l3/bias:0critic/l3/bias/Assigncritic/l3/bias/read:02"critic/l3/bias/Initializer/zeros:08
Г
critic/dense/kernel:0critic/dense/kernel/Assigncritic/dense/kernel/read:020critic/dense/kernel/Initializer/random_uniform:08
r
critic/dense/bias:0critic/dense/bias/Assigncritic/dense/bias/read:02%critic/dense/bias/Initializer/zeros:08
p
critic/beta1_power:0critic/beta1_power/Assigncritic/beta1_power/read:02"critic/beta1_power/initial_value:0
p
critic/beta2_power:0critic/beta2_power/Assigncritic/beta2_power/read:02"critic/beta2_power/initial_value:0
И
critic/critic/w1_s/Adam:0critic/critic/w1_s/Adam/Assigncritic/critic/w1_s/Adam/read:02+critic/critic/w1_s/Adam/Initializer/zeros:0
Р
critic/critic/w1_s/Adam_1:0 critic/critic/w1_s/Adam_1/Assign critic/critic/w1_s/Adam_1/read:02-critic/critic/w1_s/Adam_1/Initializer/zeros:0
А
critic/critic/b1/Adam:0critic/critic/b1/Adam/Assigncritic/critic/b1/Adam/read:02)critic/critic/b1/Adam/Initializer/zeros:0
И
critic/critic/b1/Adam_1:0critic/critic/b1/Adam_1/Assigncritic/critic/b1/Adam_1/read:02+critic/critic/b1/Adam_1/Initializer/zeros:0
Ь
critic/critic/l2/kernel/Adam:0#critic/critic/l2/kernel/Adam/Assign#critic/critic/l2/kernel/Adam/read:020critic/critic/l2/kernel/Adam/Initializer/zeros:0
д
 critic/critic/l2/kernel/Adam_1:0%critic/critic/l2/kernel/Adam_1/Assign%critic/critic/l2/kernel/Adam_1/read:022critic/critic/l2/kernel/Adam_1/Initializer/zeros:0
Ф
critic/critic/l2/bias/Adam:0!critic/critic/l2/bias/Adam/Assign!critic/critic/l2/bias/Adam/read:02.critic/critic/l2/bias/Adam/Initializer/zeros:0
Ь
critic/critic/l2/bias/Adam_1:0#critic/critic/l2/bias/Adam_1/Assign#critic/critic/l2/bias/Adam_1/read:020critic/critic/l2/bias/Adam_1/Initializer/zeros:0
Ь
critic/critic/l3/kernel/Adam:0#critic/critic/l3/kernel/Adam/Assign#critic/critic/l3/kernel/Adam/read:020critic/critic/l3/kernel/Adam/Initializer/zeros:0
д
 critic/critic/l3/kernel/Adam_1:0%critic/critic/l3/kernel/Adam_1/Assign%critic/critic/l3/kernel/Adam_1/read:022critic/critic/l3/kernel/Adam_1/Initializer/zeros:0
Ф
critic/critic/l3/bias/Adam:0!critic/critic/l3/bias/Adam/Assign!critic/critic/l3/bias/Adam/read:02.critic/critic/l3/bias/Adam/Initializer/zeros:0
Ь
critic/critic/l3/bias/Adam_1:0#critic/critic/l3/bias/Adam_1/Assign#critic/critic/l3/bias/Adam_1/read:020critic/critic/l3/bias/Adam_1/Initializer/zeros:0
и
!critic/critic/dense/kernel/Adam:0&critic/critic/dense/kernel/Adam/Assign&critic/critic/dense/kernel/Adam/read:023critic/critic/dense/kernel/Adam/Initializer/zeros:0
░
#critic/critic/dense/kernel/Adam_1:0(critic/critic/dense/kernel/Adam_1/Assign(critic/critic/dense/kernel/Adam_1/read:025critic/critic/dense/kernel/Adam_1/Initializer/zeros:0
а
critic/critic/dense/bias/Adam:0$critic/critic/dense/bias/Adam/Assign$critic/critic/dense/bias/Adam/read:021critic/critic/dense/bias/Adam/Initializer/zeros:0
и
!critic/critic/dense/bias/Adam_1:0&critic/critic/dense/bias/Adam_1/Assign&critic/critic/dense/bias/Adam_1/read:023critic/critic/dense/bias/Adam_1/Initializer/zeros:0
g
pi/l1/kernel:0pi/l1/kernel/Assignpi/l1/kernel/read:02)pi/l1/kernel/Initializer/random_uniform:08
V
pi/l1/bias:0pi/l1/bias/Assignpi/l1/bias/read:02pi/l1/bias/Initializer/zeros:08
g
pi/l2/kernel:0pi/l2/kernel/Assignpi/l2/kernel/read:02)pi/l2/kernel/Initializer/random_uniform:08
V
pi/l2/bias:0pi/l2/bias/Assignpi/l2/bias/read:02pi/l2/bias/Initializer/zeros:08
g
pi/l3/kernel:0pi/l3/kernel/Assignpi/l3/kernel/read:02)pi/l3/kernel/Initializer/random_uniform:08
V
pi/l3/bias:0pi/l3/bias/Assignpi/l3/bias/read:02pi/l3/bias/Initializer/zeros:08
g
pi/l4/kernel:0pi/l4/kernel/Assignpi/l4/kernel/read:02)pi/l4/kernel/Initializer/random_uniform:08
V
pi/l4/bias:0pi/l4/bias/Assignpi/l4/bias/read:02pi/l4/bias/Initializer/zeros:08
c
pi/a/kernel:0pi/a/kernel/Assignpi/a/kernel/read:02(pi/a/kernel/Initializer/random_uniform:08
R
pi/a/bias:0pi/a/bias/Assignpi/a/bias/read:02pi/a/bias/Initializer/zeros:08
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
q
oldpi/l1/kernel:0oldpi/l1/kernel/Assignoldpi/l1/kernel/read:02,oldpi/l1/kernel/Initializer/random_uniform:0
`
oldpi/l1/bias:0oldpi/l1/bias/Assignoldpi/l1/bias/read:02!oldpi/l1/bias/Initializer/zeros:0
q
oldpi/l2/kernel:0oldpi/l2/kernel/Assignoldpi/l2/kernel/read:02,oldpi/l2/kernel/Initializer/random_uniform:0
`
oldpi/l2/bias:0oldpi/l2/bias/Assignoldpi/l2/bias/read:02!oldpi/l2/bias/Initializer/zeros:0
q
oldpi/l3/kernel:0oldpi/l3/kernel/Assignoldpi/l3/kernel/read:02,oldpi/l3/kernel/Initializer/random_uniform:0
`
oldpi/l3/bias:0oldpi/l3/bias/Assignoldpi/l3/bias/read:02!oldpi/l3/bias/Initializer/zeros:0
q
oldpi/l4/kernel:0oldpi/l4/kernel/Assignoldpi/l4/kernel/read:02,oldpi/l4/kernel/Initializer/random_uniform:0
`
oldpi/l4/bias:0oldpi/l4/bias/Assignoldpi/l4/bias/read:02!oldpi/l4/bias/Initializer/zeros:0
m
oldpi/a/kernel:0oldpi/a/kernel/Assignoldpi/a/kernel/read:02+oldpi/a/kernel/Initializer/random_uniform:0
\
oldpi/a/bias:0oldpi/a/bias/Assignoldpi/a/bias/read:02 oldpi/a/bias/Initializer/zeros:0
}
oldpi/dense/kernel:0oldpi/dense/kernel/Assignoldpi/dense/kernel/read:02/oldpi/dense/kernel/Initializer/random_uniform:0
l
oldpi/dense/bias:0oldpi/dense/bias/Assignoldpi/dense/bias/read:02$oldpi/dense/bias/Initializer/zeros:0
p
atrain/beta1_power:0atrain/beta1_power/Assignatrain/beta1_power/read:02"atrain/beta1_power/initial_value:0
p
atrain/beta2_power:0atrain/beta2_power/Assignatrain/beta2_power/read:02"atrain/beta2_power/initial_value:0
М
atrain/pi/l1/kernel/Adam:0atrain/pi/l1/kernel/Adam/Assignatrain/pi/l1/kernel/Adam/read:02,atrain/pi/l1/kernel/Adam/Initializer/zeros:0
Ф
atrain/pi/l1/kernel/Adam_1:0!atrain/pi/l1/kernel/Adam_1/Assign!atrain/pi/l1/kernel/Adam_1/read:02.atrain/pi/l1/kernel/Adam_1/Initializer/zeros:0
Д
atrain/pi/l1/bias/Adam:0atrain/pi/l1/bias/Adam/Assignatrain/pi/l1/bias/Adam/read:02*atrain/pi/l1/bias/Adam/Initializer/zeros:0
М
atrain/pi/l1/bias/Adam_1:0atrain/pi/l1/bias/Adam_1/Assignatrain/pi/l1/bias/Adam_1/read:02,atrain/pi/l1/bias/Adam_1/Initializer/zeros:0
М
atrain/pi/l2/kernel/Adam:0atrain/pi/l2/kernel/Adam/Assignatrain/pi/l2/kernel/Adam/read:02,atrain/pi/l2/kernel/Adam/Initializer/zeros:0
Ф
atrain/pi/l2/kernel/Adam_1:0!atrain/pi/l2/kernel/Adam_1/Assign!atrain/pi/l2/kernel/Adam_1/read:02.atrain/pi/l2/kernel/Adam_1/Initializer/zeros:0
Д
atrain/pi/l2/bias/Adam:0atrain/pi/l2/bias/Adam/Assignatrain/pi/l2/bias/Adam/read:02*atrain/pi/l2/bias/Adam/Initializer/zeros:0
М
atrain/pi/l2/bias/Adam_1:0atrain/pi/l2/bias/Adam_1/Assignatrain/pi/l2/bias/Adam_1/read:02,atrain/pi/l2/bias/Adam_1/Initializer/zeros:0
М
atrain/pi/l3/kernel/Adam:0atrain/pi/l3/kernel/Adam/Assignatrain/pi/l3/kernel/Adam/read:02,atrain/pi/l3/kernel/Adam/Initializer/zeros:0
Ф
atrain/pi/l3/kernel/Adam_1:0!atrain/pi/l3/kernel/Adam_1/Assign!atrain/pi/l3/kernel/Adam_1/read:02.atrain/pi/l3/kernel/Adam_1/Initializer/zeros:0
Д
atrain/pi/l3/bias/Adam:0atrain/pi/l3/bias/Adam/Assignatrain/pi/l3/bias/Adam/read:02*atrain/pi/l3/bias/Adam/Initializer/zeros:0
М
atrain/pi/l3/bias/Adam_1:0atrain/pi/l3/bias/Adam_1/Assignatrain/pi/l3/bias/Adam_1/read:02,atrain/pi/l3/bias/Adam_1/Initializer/zeros:0
М
atrain/pi/l4/kernel/Adam:0atrain/pi/l4/kernel/Adam/Assignatrain/pi/l4/kernel/Adam/read:02,atrain/pi/l4/kernel/Adam/Initializer/zeros:0
Ф
atrain/pi/l4/kernel/Adam_1:0!atrain/pi/l4/kernel/Adam_1/Assign!atrain/pi/l4/kernel/Adam_1/read:02.atrain/pi/l4/kernel/Adam_1/Initializer/zeros:0
Д
atrain/pi/l4/bias/Adam:0atrain/pi/l4/bias/Adam/Assignatrain/pi/l4/bias/Adam/read:02*atrain/pi/l4/bias/Adam/Initializer/zeros:0
М
atrain/pi/l4/bias/Adam_1:0atrain/pi/l4/bias/Adam_1/Assignatrain/pi/l4/bias/Adam_1/read:02,atrain/pi/l4/bias/Adam_1/Initializer/zeros:0
И
atrain/pi/a/kernel/Adam:0atrain/pi/a/kernel/Adam/Assignatrain/pi/a/kernel/Adam/read:02+atrain/pi/a/kernel/Adam/Initializer/zeros:0
Р
atrain/pi/a/kernel/Adam_1:0 atrain/pi/a/kernel/Adam_1/Assign atrain/pi/a/kernel/Adam_1/read:02-atrain/pi/a/kernel/Adam_1/Initializer/zeros:0
А
atrain/pi/a/bias/Adam:0atrain/pi/a/bias/Adam/Assignatrain/pi/a/bias/Adam/read:02)atrain/pi/a/bias/Adam/Initializer/zeros:0
И
atrain/pi/a/bias/Adam_1:0atrain/pi/a/bias/Adam_1/Assignatrain/pi/a/bias/Adam_1/read:02+atrain/pi/a/bias/Adam_1/Initializer/zeros:0
Ш
atrain/pi/dense/kernel/Adam:0"atrain/pi/dense/kernel/Adam/Assign"atrain/pi/dense/kernel/Adam/read:02/atrain/pi/dense/kernel/Adam/Initializer/zeros:0
а
atrain/pi/dense/kernel/Adam_1:0$atrain/pi/dense/kernel/Adam_1/Assign$atrain/pi/dense/kernel/Adam_1/read:021atrain/pi/dense/kernel/Adam_1/Initializer/zeros:0
Р
atrain/pi/dense/bias/Adam:0 atrain/pi/dense/bias/Adam/Assign atrain/pi/dense/bias/Adam/read:02-atrain/pi/dense/bias/Adam/Initializer/zeros:0
Ш
atrain/pi/dense/bias/Adam_1:0"atrain/pi/dense/bias/Adam_1/Assign"atrain/pi/dense/bias/Adam_1/read:02/atrain/pi/dense/bias/Adam_1/Initializer/zeros:0"п
trainable_variablesЧФ
c
critic/w1_s:0critic/w1_s/Assigncritic/w1_s/read:02(critic/w1_s/Initializer/random_uniform:08
[
critic/b1:0critic/b1/Assigncritic/b1/read:02&critic/b1/Initializer/random_uniform:08
w
critic/l2/kernel:0critic/l2/kernel/Assigncritic/l2/kernel/read:02-critic/l2/kernel/Initializer/random_uniform:08
f
critic/l2/bias:0critic/l2/bias/Assigncritic/l2/bias/read:02"critic/l2/bias/Initializer/zeros:08
w
critic/l3/kernel:0critic/l3/kernel/Assigncritic/l3/kernel/read:02-critic/l3/kernel/Initializer/random_uniform:08
f
critic/l3/bias:0critic/l3/bias/Assigncritic/l3/bias/read:02"critic/l3/bias/Initializer/zeros:08
Г
critic/dense/kernel:0critic/dense/kernel/Assigncritic/dense/kernel/read:020critic/dense/kernel/Initializer/random_uniform:08
r
critic/dense/bias:0critic/dense/bias/Assigncritic/dense/bias/read:02%critic/dense/bias/Initializer/zeros:08
g
pi/l1/kernel:0pi/l1/kernel/Assignpi/l1/kernel/read:02)pi/l1/kernel/Initializer/random_uniform:08
V
pi/l1/bias:0pi/l1/bias/Assignpi/l1/bias/read:02pi/l1/bias/Initializer/zeros:08
g
pi/l2/kernel:0pi/l2/kernel/Assignpi/l2/kernel/read:02)pi/l2/kernel/Initializer/random_uniform:08
V
pi/l2/bias:0pi/l2/bias/Assignpi/l2/bias/read:02pi/l2/bias/Initializer/zeros:08
g
pi/l3/kernel:0pi/l3/kernel/Assignpi/l3/kernel/read:02)pi/l3/kernel/Initializer/random_uniform:08
V
pi/l3/bias:0pi/l3/bias/Assignpi/l3/bias/read:02pi/l3/bias/Initializer/zeros:08
g
pi/l4/kernel:0pi/l4/kernel/Assignpi/l4/kernel/read:02)pi/l4/kernel/Initializer/random_uniform:08
V
pi/l4/bias:0pi/l4/bias/Assignpi/l4/bias/read:02pi/l4/bias/Initializer/zeros:08
c
pi/a/kernel:0pi/a/kernel/Assignpi/a/kernel/read:02(pi/a/kernel/Initializer/random_uniform:08
R
pi/a/bias:0pi/a/bias/Assignpi/a/bias/read:02pi/a/bias/Initializer/zeros:08
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08gЫ