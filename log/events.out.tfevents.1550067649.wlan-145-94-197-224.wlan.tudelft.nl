       ŁK"	  @đ	×Abrain.Event:2QËŹ[ž     čh-	ľqđ	×A"Îü

h
statePlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙*
dtype0

,critic/w1_s/Initializer/random_uniform/shapeConst*
_class
loc:@critic/w1_s*
valueB"      *
dtype0*
_output_shapes
:

*critic/w1_s/Initializer/random_uniform/minConst*
_class
loc:@critic/w1_s*
valueB
 *°îž*
dtype0*
_output_shapes
: 

*critic/w1_s/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
_class
loc:@critic/w1_s*
valueB
 *°î>
ă
4critic/w1_s/Initializer/random_uniform/RandomUniformRandomUniform,critic/w1_s/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@critic/w1_s*
seed2 *
dtype0*
_output_shapes
:	
Ę
*critic/w1_s/Initializer/random_uniform/subSub*critic/w1_s/Initializer/random_uniform/max*critic/w1_s/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@critic/w1_s
Ý
*critic/w1_s/Initializer/random_uniform/mulMul4critic/w1_s/Initializer/random_uniform/RandomUniform*critic/w1_s/Initializer/random_uniform/sub*
_class
loc:@critic/w1_s*
_output_shapes
:	*
T0
Ď
&critic/w1_s/Initializer/random_uniformAdd*critic/w1_s/Initializer/random_uniform/mul*critic/w1_s/Initializer/random_uniform/min*
T0*
_class
loc:@critic/w1_s*
_output_shapes
:	
Ą
critic/w1_s
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *
_class
loc:@critic/w1_s*
	container 
Ä
critic/w1_s/AssignAssigncritic/w1_s&critic/w1_s/Initializer/random_uniform*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@critic/w1_s*
validate_shape(
s
critic/w1_s/readIdentitycritic/w1_s*
_output_shapes
:	*
T0*
_class
loc:@critic/w1_s

*critic/b1/Initializer/random_uniform/shapeConst*
_class
loc:@critic/b1*
valueB"      *
dtype0*
_output_shapes
:

(critic/b1/Initializer/random_uniform/minConst*
_class
loc:@critic/b1*
valueB
 *Ivž*
dtype0*
_output_shapes
: 

(critic/b1/Initializer/random_uniform/maxConst*
_class
loc:@critic/b1*
valueB
 *Iv>*
dtype0*
_output_shapes
: 
Ý
2critic/b1/Initializer/random_uniform/RandomUniformRandomUniform*critic/b1/Initializer/random_uniform/shape*
T0*
_class
loc:@critic/b1*
seed2 *
dtype0*
_output_shapes
:	*

seed 
Â
(critic/b1/Initializer/random_uniform/subSub(critic/b1/Initializer/random_uniform/max(critic/b1/Initializer/random_uniform/min*
T0*
_class
loc:@critic/b1*
_output_shapes
: 
Ő
(critic/b1/Initializer/random_uniform/mulMul2critic/b1/Initializer/random_uniform/RandomUniform(critic/b1/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*
_class
loc:@critic/b1
Ç
$critic/b1/Initializer/random_uniformAdd(critic/b1/Initializer/random_uniform/mul(critic/b1/Initializer/random_uniform/min*
_output_shapes
:	*
T0*
_class
loc:@critic/b1

	critic/b1
VariableV2*
_output_shapes
:	*
shared_name *
_class
loc:@critic/b1*
	container *
shape:	*
dtype0
ź
critic/b1/AssignAssign	critic/b1$critic/b1/Initializer/random_uniform*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@critic/b1*
validate_shape(
m
critic/b1/readIdentity	critic/b1*
T0*
_class
loc:@critic/b1*
_output_shapes
:	

critic/MatMulMatMulstatecritic/w1_s/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
c

critic/addAddcritic/MatMulcritic/b1/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
critic/ReluRelu
critic/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
1critic/l2/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@critic/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:

/critic/l2/kernel/Initializer/random_uniform/minConst*#
_class
loc:@critic/l2/kernel*
valueB
 *×łÝ˝*
dtype0*
_output_shapes
: 

/critic/l2/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@critic/l2/kernel*
valueB
 *×łÝ=*
dtype0*
_output_shapes
: 
ó
9critic/l2/kernel/Initializer/random_uniform/RandomUniformRandomUniform1critic/l2/kernel/Initializer/random_uniform/shape*
T0*#
_class
loc:@critic/l2/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
Ţ
/critic/l2/kernel/Initializer/random_uniform/subSub/critic/l2/kernel/Initializer/random_uniform/max/critic/l2/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@critic/l2/kernel*
_output_shapes
: 
ň
/critic/l2/kernel/Initializer/random_uniform/mulMul9critic/l2/kernel/Initializer/random_uniform/RandomUniform/critic/l2/kernel/Initializer/random_uniform/sub*#
_class
loc:@critic/l2/kernel* 
_output_shapes
:
*
T0
ä
+critic/l2/kernel/Initializer/random_uniformAdd/critic/l2/kernel/Initializer/random_uniform/mul/critic/l2/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@critic/l2/kernel* 
_output_shapes
:

­
critic/l2/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *#
_class
loc:@critic/l2/kernel*
	container *
shape:

Ů
critic/l2/kernel/AssignAssigncritic/l2/kernel+critic/l2/kernel/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@critic/l2/kernel*
validate_shape(

critic/l2/kernel/readIdentitycritic/l2/kernel*#
_class
loc:@critic/l2/kernel* 
_output_shapes
:
*
T0

 critic/l2/bias/Initializer/zerosConst*!
_class
loc:@critic/l2/bias*
valueB*    *
dtype0*
_output_shapes	
:

critic/l2/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *!
_class
loc:@critic/l2/bias
Ă
critic/l2/bias/AssignAssigncritic/l2/bias critic/l2/bias/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@critic/l2/bias*
validate_shape(*
_output_shapes	
:
x
critic/l2/bias/readIdentitycritic/l2/bias*!
_class
loc:@critic/l2/bias*
_output_shapes	
:*
T0

critic/l2/MatMulMatMulcritic/Relucritic/l2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

critic/l2/BiasAddBiasAddcritic/l2/MatMulcritic/l2/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
\
critic/l2/ReluRelucritic/l2/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
§
1critic/l3/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@critic/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:

/critic/l3/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *#
_class
loc:@critic/l3/kernel*
valueB
 *   ž

/critic/l3/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@critic/l3/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
ó
9critic/l3/kernel/Initializer/random_uniform/RandomUniformRandomUniform1critic/l3/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*#
_class
loc:@critic/l3/kernel*
seed2 
Ţ
/critic/l3/kernel/Initializer/random_uniform/subSub/critic/l3/kernel/Initializer/random_uniform/max/critic/l3/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@critic/l3/kernel*
_output_shapes
: 
ň
/critic/l3/kernel/Initializer/random_uniform/mulMul9critic/l3/kernel/Initializer/random_uniform/RandomUniform/critic/l3/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*#
_class
loc:@critic/l3/kernel
ä
+critic/l3/kernel/Initializer/random_uniformAdd/critic/l3/kernel/Initializer/random_uniform/mul/critic/l3/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:

­
critic/l3/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *#
_class
loc:@critic/l3/kernel*
	container 
Ů
critic/l3/kernel/AssignAssigncritic/l3/kernel+critic/l3/kernel/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@critic/l3/kernel*
validate_shape(

critic/l3/kernel/readIdentitycritic/l3/kernel*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:


 critic/l3/bias/Initializer/zerosConst*!
_class
loc:@critic/l3/bias*
valueB*    *
dtype0*
_output_shapes	
:

critic/l3/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *!
_class
loc:@critic/l3/bias*
	container *
shape:
Ă
critic/l3/bias/AssignAssigncritic/l3/bias critic/l3/bias/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@critic/l3/bias*
validate_shape(*
_output_shapes	
:
x
critic/l3/bias/readIdentitycritic/l3/bias*
T0*!
_class
loc:@critic/l3/bias*
_output_shapes	
:

critic/l3/MatMulMatMulcritic/l2/Relucritic/l3/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

critic/l3/BiasAddBiasAddcritic/l3/MatMulcritic/l3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
critic/l3/ReluRelucritic/l3/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
4critic/dense/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@critic/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:

2critic/dense/kernel/Initializer/random_uniform/minConst*&
_class
loc:@critic/dense/kernel*
valueB
 *n×\ž*
dtype0*
_output_shapes
: 

2critic/dense/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@critic/dense/kernel*
valueB
 *n×\>*
dtype0*
_output_shapes
: 
ű
<critic/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform4critic/dense/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@critic/dense/kernel*
seed2 *
dtype0*
_output_shapes
:	
ę
2critic/dense/kernel/Initializer/random_uniform/subSub2critic/dense/kernel/Initializer/random_uniform/max2critic/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*&
_class
loc:@critic/dense/kernel
ý
2critic/dense/kernel/Initializer/random_uniform/mulMul<critic/dense/kernel/Initializer/random_uniform/RandomUniform2critic/dense/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	
ď
.critic/dense/kernel/Initializer/random_uniformAdd2critic/dense/kernel/Initializer/random_uniform/mul2critic/dense/kernel/Initializer/random_uniform/min*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	*
T0
ą
critic/dense/kernel
VariableV2*
shared_name *&
_class
loc:@critic/dense/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
ä
critic/dense/kernel/AssignAssigncritic/dense/kernel.critic/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*&
_class
loc:@critic/dense/kernel*
validate_shape(*
_output_shapes
:	

critic/dense/kernel/readIdentitycritic/dense/kernel*
_output_shapes
:	*
T0*&
_class
loc:@critic/dense/kernel

#critic/dense/bias/Initializer/zerosConst*$
_class
loc:@critic/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Ł
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
Î
critic/dense/bias/AssignAssigncritic/dense/bias#critic/dense/bias/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@critic/dense/bias*
validate_shape(*
_output_shapes
:

critic/dense/bias/readIdentitycritic/dense/bias*
T0*$
_class
loc:@critic/dense/bias*
_output_shapes
:

critic/dense/MatMulMatMulcritic/l3/Relucritic/dense/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

critic/dense/BiasAddBiasAddcritic/dense/MatMulcritic/dense/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
v
critic/discounted_rPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙*
dtype0
n

critic/subSubcritic/discounted_rcritic/dense/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
U
critic/SquareSquare
critic/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
critic/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
n
critic/MeanMeancritic/Squarecritic/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Y
critic/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
_
critic/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  ?*
dtype0

critic/gradients/FillFillcritic/gradients/Shapecritic/gradients/grad_ys_0*

index_type0*
_output_shapes
: *
T0

/critic/gradients/critic/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ł
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
Ć
&critic/gradients/critic/Mean_grad/TileTile)critic/gradients/critic/Mean_grad/Reshape'critic/gradients/critic/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
'critic/gradients/critic/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ŕ
&critic/gradients/critic/Mean_grad/ProdProd)critic/gradients/critic/Mean_grad/Shape_1'critic/gradients/critic/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
s
)critic/gradients/critic/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
Ä
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
Ź
)critic/gradients/critic/Mean_grad/MaximumMaximum(critic/gradients/critic/Mean_grad/Prod_1+critic/gradients/critic/Mean_grad/Maximum/y*
_output_shapes
: *
T0
Ş
*critic/gradients/critic/Mean_grad/floordivFloorDiv&critic/gradients/critic/Mean_grad/Prod)critic/gradients/critic/Mean_grad/Maximum*
_output_shapes
: *
T0

&critic/gradients/critic/Mean_grad/CastCast*critic/gradients/critic/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
ś
)critic/gradients/critic/Mean_grad/truedivRealDiv&critic/gradients/critic/Mean_grad/Tile&critic/gradients/critic/Mean_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

)critic/gradients/critic/Square_grad/ConstConst*^critic/gradients/critic/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 

'critic/gradients/critic/Square_grad/MulMul
critic/sub)critic/gradients/critic/Square_grad/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
)critic/gradients/critic/Square_grad/Mul_1Mul)critic/gradients/critic/Mean_grad/truediv'critic/gradients/critic/Square_grad/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
Ţ
6critic/gradients/critic/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&critic/gradients/critic/sub_grad/Shape(critic/gradients/critic/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Î
$critic/gradients/critic/sub_grad/SumSum)critic/gradients/critic/Square_grad/Mul_16critic/gradients/critic/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Á
(critic/gradients/critic/sub_grad/ReshapeReshape$critic/gradients/critic/sub_grad/Sum&critic/gradients/critic/sub_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ň
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
Ĺ
*critic/gradients/critic/sub_grad/Reshape_1Reshape$critic/gradients/critic/sub_grad/Neg(critic/gradients/critic/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

1critic/gradients/critic/sub_grad/tuple/group_depsNoOp)^critic/gradients/critic/sub_grad/Reshape+^critic/gradients/critic/sub_grad/Reshape_1

9critic/gradients/critic/sub_grad/tuple/control_dependencyIdentity(critic/gradients/critic/sub_grad/Reshape2^critic/gradients/critic/sub_grad/tuple/group_deps*;
_class1
/-loc:@critic/gradients/critic/sub_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

;critic/gradients/critic/sub_grad/tuple/control_dependency_1Identity*critic/gradients/critic/sub_grad/Reshape_12^critic/gradients/critic/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@critic/gradients/critic/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
6critic/gradients/critic/dense/BiasAdd_grad/BiasAddGradBiasAddGrad;critic/gradients/critic/sub_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:
ş
;critic/gradients/critic/dense/BiasAdd_grad/tuple/group_depsNoOp7^critic/gradients/critic/dense/BiasAdd_grad/BiasAddGrad<^critic/gradients/critic/sub_grad/tuple/control_dependency_1
ť
Ccritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependencyIdentity;critic/gradients/critic/sub_grad/tuple/control_dependency_1<^critic/gradients/critic/dense/BiasAdd_grad/tuple/group_deps*=
_class3
1/loc:@critic/gradients/critic/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ˇ
Ecritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependency_1Identity6critic/gradients/critic/dense/BiasAdd_grad/BiasAddGrad<^critic/gradients/critic/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*I
_class?
=;loc:@critic/gradients/critic/dense/BiasAdd_grad/BiasAddGrad
ň
0critic/gradients/critic/dense/MatMul_grad/MatMulMatMulCcritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependencycritic/dense/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
á
2critic/gradients/critic/dense/MatMul_grad/MatMul_1MatMulcritic/l3/ReluCcritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0
Ş
:critic/gradients/critic/dense/MatMul_grad/tuple/group_depsNoOp1^critic/gradients/critic/dense/MatMul_grad/MatMul3^critic/gradients/critic/dense/MatMul_grad/MatMul_1
ľ
Bcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependencyIdentity0critic/gradients/critic/dense/MatMul_grad/MatMul;^critic/gradients/critic/dense/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@critic/gradients/critic/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
Dcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependency_1Identity2critic/gradients/critic/dense/MatMul_grad/MatMul_1;^critic/gradients/critic/dense/MatMul_grad/tuple/group_deps*E
_class;
97loc:@critic/gradients/critic/dense/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
Ŕ
-critic/gradients/critic/l3/Relu_grad/ReluGradReluGradBcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependencycritic/l3/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
3critic/gradients/critic/l3/BiasAdd_grad/BiasAddGradBiasAddGrad-critic/gradients/critic/l3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ś
8critic/gradients/critic/l3/BiasAdd_grad/tuple/group_depsNoOp4^critic/gradients/critic/l3/BiasAdd_grad/BiasAddGrad.^critic/gradients/critic/l3/Relu_grad/ReluGrad
Ť
@critic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l3/Relu_grad/ReluGrad9^critic/gradients/critic/l3/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@critic/gradients/critic/l3/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
Bcritic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependency_1Identity3critic/gradients/critic/l3/BiasAdd_grad/BiasAddGrad9^critic/gradients/critic/l3/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*F
_class<
:8loc:@critic/gradients/critic/l3/BiasAdd_grad/BiasAddGrad
é
-critic/gradients/critic/l3/MatMul_grad/MatMulMatMul@critic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependencycritic/l3/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ü
/critic/gradients/critic/l3/MatMul_grad/MatMul_1MatMulcritic/l2/Relu@critic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
Ą
7critic/gradients/critic/l3/MatMul_grad/tuple/group_depsNoOp.^critic/gradients/critic/l3/MatMul_grad/MatMul0^critic/gradients/critic/l3/MatMul_grad/MatMul_1
Š
?critic/gradients/critic/l3/MatMul_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l3/MatMul_grad/MatMul8^critic/gradients/critic/l3/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@critic/gradients/critic/l3/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
Acritic/gradients/critic/l3/MatMul_grad/tuple/control_dependency_1Identity/critic/gradients/critic/l3/MatMul_grad/MatMul_18^critic/gradients/critic/l3/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@critic/gradients/critic/l3/MatMul_grad/MatMul_1* 
_output_shapes
:

˝
-critic/gradients/critic/l2/Relu_grad/ReluGradReluGrad?critic/gradients/critic/l3/MatMul_grad/tuple/control_dependencycritic/l2/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
3critic/gradients/critic/l2/BiasAdd_grad/BiasAddGradBiasAddGrad-critic/gradients/critic/l2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
Ś
8critic/gradients/critic/l2/BiasAdd_grad/tuple/group_depsNoOp4^critic/gradients/critic/l2/BiasAdd_grad/BiasAddGrad.^critic/gradients/critic/l2/Relu_grad/ReluGrad
Ť
@critic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l2/Relu_grad/ReluGrad9^critic/gradients/critic/l2/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@critic/gradients/critic/l2/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
Bcritic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependency_1Identity3critic/gradients/critic/l2/BiasAdd_grad/BiasAddGrad9^critic/gradients/critic/l2/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@critic/gradients/critic/l2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
é
-critic/gradients/critic/l2/MatMul_grad/MatMulMatMul@critic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependencycritic/l2/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ů
/critic/gradients/critic/l2/MatMul_grad/MatMul_1MatMulcritic/Relu@critic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
Ą
7critic/gradients/critic/l2/MatMul_grad/tuple/group_depsNoOp.^critic/gradients/critic/l2/MatMul_grad/MatMul0^critic/gradients/critic/l2/MatMul_grad/MatMul_1
Š
?critic/gradients/critic/l2/MatMul_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l2/MatMul_grad/MatMul8^critic/gradients/critic/l2/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@critic/gradients/critic/l2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
Acritic/gradients/critic/l2/MatMul_grad/tuple/control_dependency_1Identity/critic/gradients/critic/l2/MatMul_grad/MatMul_18^critic/gradients/critic/l2/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@critic/gradients/critic/l2/MatMul_grad/MatMul_1* 
_output_shapes
:

ˇ
*critic/gradients/critic/Relu_grad/ReluGradReluGrad?critic/gradients/critic/l2/MatMul_grad/tuple/control_dependencycritic/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
Ţ
6critic/gradients/critic/add_grad/BroadcastGradientArgsBroadcastGradientArgs&critic/gradients/critic/add_grad/Shape(critic/gradients/critic/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ď
$critic/gradients/critic/add_grad/SumSum*critic/gradients/critic/Relu_grad/ReluGrad6critic/gradients/critic/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Â
(critic/gradients/critic/add_grad/ReshapeReshape$critic/gradients/critic/add_grad/Sum&critic/gradients/critic/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
&critic/gradients/critic/add_grad/Sum_1Sum*critic/gradients/critic/Relu_grad/ReluGrad8critic/gradients/critic/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ż
*critic/gradients/critic/add_grad/Reshape_1Reshape&critic/gradients/critic/add_grad/Sum_1(critic/gradients/critic/add_grad/Shape_1*
_output_shapes
:	*
T0*
Tshape0

1critic/gradients/critic/add_grad/tuple/group_depsNoOp)^critic/gradients/critic/add_grad/Reshape+^critic/gradients/critic/add_grad/Reshape_1

9critic/gradients/critic/add_grad/tuple/control_dependencyIdentity(critic/gradients/critic/add_grad/Reshape2^critic/gradients/critic/add_grad/tuple/group_deps*;
_class1
/-loc:@critic/gradients/critic/add_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

;critic/gradients/critic/add_grad/tuple/control_dependency_1Identity*critic/gradients/critic/add_grad/Reshape_12^critic/gradients/critic/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@critic/gradients/critic/add_grad/Reshape_1*
_output_shapes
:	
Ů
*critic/gradients/critic/MatMul_grad/MatMulMatMul9critic/gradients/critic/add_grad/tuple/control_dependencycritic/w1_s/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Č
,critic/gradients/critic/MatMul_grad/MatMul_1MatMulstate9critic/gradients/critic/add_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 

4critic/gradients/critic/MatMul_grad/tuple/group_depsNoOp+^critic/gradients/critic/MatMul_grad/MatMul-^critic/gradients/critic/MatMul_grad/MatMul_1

<critic/gradients/critic/MatMul_grad/tuple/control_dependencyIdentity*critic/gradients/critic/MatMul_grad/MatMul5^critic/gradients/critic/MatMul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*=
_class3
1/loc:@critic/gradients/critic/MatMul_grad/MatMul

>critic/gradients/critic/MatMul_grad/tuple/control_dependency_1Identity,critic/gradients/critic/MatMul_grad/MatMul_15^critic/gradients/critic/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@critic/gradients/critic/MatMul_grad/MatMul_1*
_output_shapes
:	

 critic/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@critic/b1*
valueB
 *fff?

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
Á
critic/beta1_power/AssignAssigncritic/beta1_power critic/beta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@critic/b1
v
critic/beta1_power/readIdentitycritic/beta1_power*
T0*
_class
loc:@critic/b1*
_output_shapes
: 

 critic/beta2_power/initial_valueConst*
_class
loc:@critic/b1*
valueB
 *wž?*
dtype0*
_output_shapes
: 

critic/beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@critic/b1*
	container 
Á
critic/beta2_power/AssignAssigncritic/beta2_power critic/beta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@critic/b1*
validate_shape(
v
critic/beta2_power/readIdentitycritic/beta2_power*
T0*
_class
loc:@critic/b1*
_output_shapes
: 
Ş
9critic/critic/w1_s/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@critic/w1_s*
valueB"      *
dtype0*
_output_shapes
:

/critic/critic/w1_s/Adam/Initializer/zeros/ConstConst*
_class
loc:@critic/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
ů
)critic/critic/w1_s/Adam/Initializer/zerosFill9critic/critic/w1_s/Adam/Initializer/zeros/shape_as_tensor/critic/critic/w1_s/Adam/Initializer/zeros/Const*
_class
loc:@critic/w1_s*

index_type0*
_output_shapes
:	*
T0
­
critic/critic/w1_s/Adam
VariableV2*
_class
loc:@critic/w1_s*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 
ß
critic/critic/w1_s/Adam/AssignAssigncritic/critic/w1_s/Adam)critic/critic/w1_s/Adam/Initializer/zeros*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@critic/w1_s*
validate_shape(

critic/critic/w1_s/Adam/readIdentitycritic/critic/w1_s/Adam*
T0*
_class
loc:@critic/w1_s*
_output_shapes
:	
Ź
;critic/critic/w1_s/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@critic/w1_s*
valueB"      *
dtype0*
_output_shapes
:

1critic/critic/w1_s/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@critic/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
˙
+critic/critic/w1_s/Adam_1/Initializer/zerosFill;critic/critic/w1_s/Adam_1/Initializer/zeros/shape_as_tensor1critic/critic/w1_s/Adam_1/Initializer/zeros/Const*
_class
loc:@critic/w1_s*

index_type0*
_output_shapes
:	*
T0
Ż
critic/critic/w1_s/Adam_1
VariableV2*
shared_name *
_class
loc:@critic/w1_s*
	container *
shape:	*
dtype0*
_output_shapes
:	
ĺ
 critic/critic/w1_s/Adam_1/AssignAssigncritic/critic/w1_s/Adam_1+critic/critic/w1_s/Adam_1/Initializer/zeros*
T0*
_class
loc:@critic/w1_s*
validate_shape(*
_output_shapes
:	*
use_locking(

critic/critic/w1_s/Adam_1/readIdentitycritic/critic/w1_s/Adam_1*
_output_shapes
:	*
T0*
_class
loc:@critic/w1_s

'critic/critic/b1/Adam/Initializer/zerosConst*
_class
loc:@critic/b1*
valueB	*    *
dtype0*
_output_shapes
:	
Š
critic/critic/b1/Adam
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *
_class
loc:@critic/b1*
	container *
shape:	
×
critic/critic/b1/Adam/AssignAssigncritic/critic/b1/Adam'critic/critic/b1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@critic/b1

critic/critic/b1/Adam/readIdentitycritic/critic/b1/Adam*
_class
loc:@critic/b1*
_output_shapes
:	*
T0

)critic/critic/b1/Adam_1/Initializer/zerosConst*
_class
loc:@critic/b1*
valueB	*    *
dtype0*
_output_shapes
:	
Ť
critic/critic/b1/Adam_1
VariableV2*
shared_name *
_class
loc:@critic/b1*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ý
critic/critic/b1/Adam_1/AssignAssigncritic/critic/b1/Adam_1)critic/critic/b1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
:	

critic/critic/b1/Adam_1/readIdentitycritic/critic/b1/Adam_1*
T0*
_class
loc:@critic/b1*
_output_shapes
:	
´
>critic/critic/l2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@critic/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:

4critic/critic/l2/kernel/Adam/Initializer/zeros/ConstConst*#
_class
loc:@critic/l2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

.critic/critic/l2/kernel/Adam/Initializer/zerosFill>critic/critic/l2/kernel/Adam/Initializer/zeros/shape_as_tensor4critic/critic/l2/kernel/Adam/Initializer/zeros/Const*#
_class
loc:@critic/l2/kernel*

index_type0* 
_output_shapes
:
*
T0
š
critic/critic/l2/kernel/Adam
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *#
_class
loc:@critic/l2/kernel*
	container 
ô
#critic/critic/l2/kernel/Adam/AssignAssigncritic/critic/l2/kernel/Adam.critic/critic/l2/kernel/Adam/Initializer/zeros*#
_class
loc:@critic/l2/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

!critic/critic/l2/kernel/Adam/readIdentitycritic/critic/l2/kernel/Adam*
T0*#
_class
loc:@critic/l2/kernel* 
_output_shapes
:

ś
@critic/critic/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*#
_class
loc:@critic/l2/kernel*
valueB"      *
dtype0
 
6critic/critic/l2/kernel/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@critic/l2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

0critic/critic/l2/kernel/Adam_1/Initializer/zerosFill@critic/critic/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensor6critic/critic/l2/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*#
_class
loc:@critic/l2/kernel*

index_type0
ť
critic/critic/l2/kernel/Adam_1
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *#
_class
loc:@critic/l2/kernel*
	container 
ú
%critic/critic/l2/kernel/Adam_1/AssignAssigncritic/critic/l2/kernel/Adam_10critic/critic/l2/kernel/Adam_1/Initializer/zeros*#
_class
loc:@critic/l2/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

#critic/critic/l2/kernel/Adam_1/readIdentitycritic/critic/l2/kernel/Adam_1* 
_output_shapes
:
*
T0*#
_class
loc:@critic/l2/kernel

,critic/critic/l2/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*!
_class
loc:@critic/l2/bias*
valueB*    
Ť
critic/critic/l2/bias/Adam
VariableV2*!
_class
loc:@critic/l2/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ç
!critic/critic/l2/bias/Adam/AssignAssigncritic/critic/l2/bias/Adam,critic/critic/l2/bias/Adam/Initializer/zeros*
T0*!
_class
loc:@critic/l2/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

critic/critic/l2/bias/Adam/readIdentitycritic/critic/l2/bias/Adam*
T0*!
_class
loc:@critic/l2/bias*
_output_shapes	
:
 
.critic/critic/l2/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@critic/l2/bias*
valueB*    *
dtype0*
_output_shapes	
:
­
critic/critic/l2/bias/Adam_1
VariableV2*
_output_shapes	
:*
shared_name *!
_class
loc:@critic/l2/bias*
	container *
shape:*
dtype0
í
#critic/critic/l2/bias/Adam_1/AssignAssigncritic/critic/l2/bias/Adam_1.critic/critic/l2/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@critic/l2/bias

!critic/critic/l2/bias/Adam_1/readIdentitycritic/critic/l2/bias/Adam_1*
_output_shapes	
:*
T0*!
_class
loc:@critic/l2/bias
´
>critic/critic/l3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*#
_class
loc:@critic/l3/kernel*
valueB"      *
dtype0

4critic/critic/l3/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *#
_class
loc:@critic/l3/kernel*
valueB
 *    

.critic/critic/l3/kernel/Adam/Initializer/zerosFill>critic/critic/l3/kernel/Adam/Initializer/zeros/shape_as_tensor4critic/critic/l3/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*
T0*#
_class
loc:@critic/l3/kernel*

index_type0
š
critic/critic/l3/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *#
_class
loc:@critic/l3/kernel*
	container *
shape:

ô
#critic/critic/l3/kernel/Adam/AssignAssigncritic/critic/l3/kernel/Adam.critic/critic/l3/kernel/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@critic/l3/kernel*
validate_shape(* 
_output_shapes
:


!critic/critic/l3/kernel/Adam/readIdentitycritic/critic/l3/kernel/Adam*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:

ś
@critic/critic/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@critic/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:
 
6critic/critic/l3/kernel/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@critic/l3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

0critic/critic/l3/kernel/Adam_1/Initializer/zerosFill@critic/critic/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensor6critic/critic/l3/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*#
_class
loc:@critic/l3/kernel*

index_type0
ť
critic/critic/l3/kernel/Adam_1
VariableV2*#
_class
loc:@critic/l3/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
ú
%critic/critic/l3/kernel/Adam_1/AssignAssigncritic/critic/l3/kernel/Adam_10critic/critic/l3/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@critic/l3/kernel*
validate_shape(

#critic/critic/l3/kernel/Adam_1/readIdentitycritic/critic/l3/kernel/Adam_1* 
_output_shapes
:
*
T0*#
_class
loc:@critic/l3/kernel

,critic/critic/l3/bias/Adam/Initializer/zerosConst*!
_class
loc:@critic/l3/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ť
critic/critic/l3/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *!
_class
loc:@critic/l3/bias*
	container *
shape:
ç
!critic/critic/l3/bias/Adam/AssignAssigncritic/critic/l3/bias/Adam,critic/critic/l3/bias/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@critic/l3/bias*
validate_shape(*
_output_shapes	
:

critic/critic/l3/bias/Adam/readIdentitycritic/critic/l3/bias/Adam*
_output_shapes	
:*
T0*!
_class
loc:@critic/l3/bias
 
.critic/critic/l3/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@critic/l3/bias*
valueB*    *
dtype0*
_output_shapes	
:
­
critic/critic/l3/bias/Adam_1
VariableV2*
shared_name *!
_class
loc:@critic/l3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
í
#critic/critic/l3/bias/Adam_1/AssignAssigncritic/critic/l3/bias/Adam_1.critic/critic/l3/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@critic/l3/bias*
validate_shape(*
_output_shapes	
:

!critic/critic/l3/bias/Adam_1/readIdentitycritic/critic/l3/bias/Adam_1*
_output_shapes	
:*
T0*!
_class
loc:@critic/l3/bias
°
1critic/critic/dense/kernel/Adam/Initializer/zerosConst*&
_class
loc:@critic/dense/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
˝
critic/critic/dense/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *&
_class
loc:@critic/dense/kernel*
	container *
shape:	
˙
&critic/critic/dense/kernel/Adam/AssignAssigncritic/critic/dense/kernel/Adam1critic/critic/dense/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*&
_class
loc:@critic/dense/kernel
Ł
$critic/critic/dense/kernel/Adam/readIdentitycritic/critic/dense/kernel/Adam*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	
˛
3critic/critic/dense/kernel/Adam_1/Initializer/zerosConst*&
_class
loc:@critic/dense/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
ż
!critic/critic/dense/kernel/Adam_1
VariableV2*
shared_name *&
_class
loc:@critic/dense/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	

(critic/critic/dense/kernel/Adam_1/AssignAssign!critic/critic/dense/kernel/Adam_13critic/critic/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@critic/dense/kernel*
validate_shape(*
_output_shapes
:	
§
&critic/critic/dense/kernel/Adam_1/readIdentity!critic/critic/dense/kernel/Adam_1*
_output_shapes
:	*
T0*&
_class
loc:@critic/dense/kernel
˘
/critic/critic/dense/bias/Adam/Initializer/zerosConst*$
_class
loc:@critic/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Ż
critic/critic/dense/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@critic/dense/bias
ň
$critic/critic/dense/bias/Adam/AssignAssigncritic/critic/dense/bias/Adam/critic/critic/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@critic/dense/bias*
validate_shape(*
_output_shapes
:

"critic/critic/dense/bias/Adam/readIdentitycritic/critic/dense/bias/Adam*
_output_shapes
:*
T0*$
_class
loc:@critic/dense/bias
¤
1critic/critic/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*$
_class
loc:@critic/dense/bias*
valueB*    *
dtype0
ą
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
ř
&critic/critic/dense/bias/Adam_1/AssignAssigncritic/critic/dense/bias/Adam_11critic/critic/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@critic/dense/bias*
validate_shape(*
_output_shapes
:

$critic/critic/dense/bias/Adam_1/readIdentitycritic/critic/dense/bias/Adam_1*
T0*$
_class
loc:@critic/dense/bias*
_output_shapes
:
^
critic/Adam/learning_rateConst*
valueB
 *ˇQ9*
dtype0*
_output_shapes
: 
V
critic/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
V
critic/Adam/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
X
critic/Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
Ż
(critic/Adam/update_critic/w1_s/ApplyAdam	ApplyAdamcritic/w1_scritic/critic/w1_s/Adamcritic/critic/w1_s/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilon>critic/gradients/critic/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@critic/w1_s*
use_nesterov( *
_output_shapes
:	
˘
&critic/Adam/update_critic/b1/ApplyAdam	ApplyAdam	critic/b1critic/critic/b1/Adamcritic/critic/b1/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilon;critic/gradients/critic/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@critic/b1*
use_nesterov( *
_output_shapes
:	
Ě
-critic/Adam/update_critic/l2/kernel/ApplyAdam	ApplyAdamcritic/l2/kernelcritic/critic/l2/kernel/Adamcritic/critic/l2/kernel/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonAcritic/gradients/critic/l2/MatMul_grad/tuple/control_dependency_1*#
_class
loc:@critic/l2/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
ž
+critic/Adam/update_critic/l2/bias/ApplyAdam	ApplyAdamcritic/l2/biascritic/critic/l2/bias/Adamcritic/critic/l2/bias/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonBcritic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*!
_class
loc:@critic/l2/bias
Ě
-critic/Adam/update_critic/l3/kernel/ApplyAdam	ApplyAdamcritic/l3/kernelcritic/critic/l3/kernel/Adamcritic/critic/l3/kernel/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonAcritic/gradients/critic/l3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@critic/l3/kernel*
use_nesterov( * 
_output_shapes
:

ž
+critic/Adam/update_critic/l3/bias/ApplyAdam	ApplyAdamcritic/l3/biascritic/critic/l3/bias/Adamcritic/critic/l3/bias/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonBcritic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependency_1*
T0*!
_class
loc:@critic/l3/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
Ý
0critic/Adam/update_critic/dense/kernel/ApplyAdam	ApplyAdamcritic/dense/kernelcritic/critic/dense/kernel/Adam!critic/critic/dense/kernel/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonDcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@critic/dense/kernel*
use_nesterov( *
_output_shapes
:	
Ď
.critic/Adam/update_critic/dense/bias/ApplyAdam	ApplyAdamcritic/dense/biascritic/critic/dense/bias/Adamcritic/critic/dense/bias/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonEcritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@critic/dense/bias*
use_nesterov( *
_output_shapes
:
ő
critic/Adam/mulMulcritic/beta1_power/readcritic/Adam/beta1'^critic/Adam/update_critic/b1/ApplyAdam/^critic/Adam/update_critic/dense/bias/ApplyAdam1^critic/Adam/update_critic/dense/kernel/ApplyAdam,^critic/Adam/update_critic/l2/bias/ApplyAdam.^critic/Adam/update_critic/l2/kernel/ApplyAdam,^critic/Adam/update_critic/l3/bias/ApplyAdam.^critic/Adam/update_critic/l3/kernel/ApplyAdam)^critic/Adam/update_critic/w1_s/ApplyAdam*
T0*
_class
loc:@critic/b1*
_output_shapes
: 
Š
critic/Adam/AssignAssigncritic/beta1_powercritic/Adam/mul*
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
: *
use_locking( 
÷
critic/Adam/mul_1Mulcritic/beta2_power/readcritic/Adam/beta2'^critic/Adam/update_critic/b1/ApplyAdam/^critic/Adam/update_critic/dense/bias/ApplyAdam1^critic/Adam/update_critic/dense/kernel/ApplyAdam,^critic/Adam/update_critic/l2/bias/ApplyAdam.^critic/Adam/update_critic/l2/kernel/ApplyAdam,^critic/Adam/update_critic/l3/bias/ApplyAdam.^critic/Adam/update_critic/l3/kernel/ApplyAdam)^critic/Adam/update_critic/w1_s/ApplyAdam*
T0*
_class
loc:@critic/b1*
_output_shapes
: 
­
critic/Adam/Assign_1Assigncritic/beta2_powercritic/Adam/mul_1*
use_locking( *
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
: 
ł
critic/AdamNoOp^critic/Adam/Assign^critic/Adam/Assign_1'^critic/Adam/update_critic/b1/ApplyAdam/^critic/Adam/update_critic/dense/bias/ApplyAdam1^critic/Adam/update_critic/dense/kernel/ApplyAdam,^critic/Adam/update_critic/l2/bias/ApplyAdam.^critic/Adam/update_critic/l2/kernel/ApplyAdam,^critic/Adam/update_critic/l3/bias/ApplyAdam.^critic/Adam/update_critic/l3/kernel/ApplyAdam)^critic/Adam/update_critic/w1_s/ApplyAdam

-pi/l1/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/l1/kernel*
valueB"      *
dtype0*
_output_shapes
:

+pi/l1/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/l1/kernel*
valueB
 *°îž*
dtype0*
_output_shapes
: 

+pi/l1/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/l1/kernel*
valueB
 *°î>*
dtype0*
_output_shapes
: 
ć
5pi/l1/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l1/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@pi/l1/kernel*
seed2 *
dtype0*
_output_shapes
:	*

seed 
Î
+pi/l1/kernel/Initializer/random_uniform/subSub+pi/l1/kernel/Initializer/random_uniform/max+pi/l1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@pi/l1/kernel
á
+pi/l1/kernel/Initializer/random_uniform/mulMul5pi/l1/kernel/Initializer/random_uniform/RandomUniform+pi/l1/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*
_class
loc:@pi/l1/kernel
Ó
'pi/l1/kernel/Initializer/random_uniformAdd+pi/l1/kernel/Initializer/random_uniform/mul+pi/l1/kernel/Initializer/random_uniform/min*
_output_shapes
:	*
T0*
_class
loc:@pi/l1/kernel
Ł
pi/l1/kernel
VariableV2*
_class
loc:@pi/l1/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 
Č
pi/l1/kernel/AssignAssignpi/l1/kernel'pi/l1/kernel/Initializer/random_uniform*
T0*
_class
loc:@pi/l1/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
v
pi/l1/kernel/readIdentitypi/l1/kernel*
_output_shapes
:	*
T0*
_class
loc:@pi/l1/kernel

pi/l1/bias/Initializer/zerosConst*
_class
loc:@pi/l1/bias*
valueB*    *
dtype0*
_output_shapes	
:


pi/l1/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l1/bias*
	container *
shape:
ł
pi/l1/bias/AssignAssign
pi/l1/biaspi/l1/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@pi/l1/bias*
validate_shape(
l
pi/l1/bias/readIdentity
pi/l1/bias*
T0*
_class
loc:@pi/l1/bias*
_output_shapes	
:

pi/l1/MatMulMatMulstatepi/l1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

pi/l1/BiasAddBiasAddpi/l1/MatMulpi/l1/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
T

pi/l1/ReluRelupi/l1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-pi/l2/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:

+pi/l2/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/l2/kernel*
valueB
 *×łÝ˝*
dtype0*
_output_shapes
: 

+pi/l2/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/l2/kernel*
valueB
 *×łÝ=*
dtype0*
_output_shapes
: 
ç
5pi/l2/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l2/kernel/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@pi/l2/kernel*
seed2 *
dtype0* 
_output_shapes
:

Î
+pi/l2/kernel/Initializer/random_uniform/subSub+pi/l2/kernel/Initializer/random_uniform/max+pi/l2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@pi/l2/kernel
â
+pi/l2/kernel/Initializer/random_uniform/mulMul5pi/l2/kernel/Initializer/random_uniform/RandomUniform+pi/l2/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@pi/l2/kernel* 
_output_shapes
:

Ô
'pi/l2/kernel/Initializer/random_uniformAdd+pi/l2/kernel/Initializer/random_uniform/mul+pi/l2/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l2/kernel* 
_output_shapes
:

Ľ
pi/l2/kernel
VariableV2*
shared_name *
_class
loc:@pi/l2/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

É
pi/l2/kernel/AssignAssignpi/l2/kernel'pi/l2/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@pi/l2/kernel*
validate_shape(* 
_output_shapes
:

w
pi/l2/kernel/readIdentitypi/l2/kernel* 
_output_shapes
:
*
T0*
_class
loc:@pi/l2/kernel

pi/l2/bias/Initializer/zerosConst*
_class
loc:@pi/l2/bias*
valueB*    *
dtype0*
_output_shapes	
:


pi/l2/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l2/bias*
	container 
ł
pi/l2/bias/AssignAssign
pi/l2/biaspi/l2/bias/Initializer/zeros*
T0*
_class
loc:@pi/l2/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
l
pi/l2/bias/readIdentity
pi/l2/bias*
_output_shapes	
:*
T0*
_class
loc:@pi/l2/bias

pi/l2/MatMulMatMul
pi/l1/Relupi/l2/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

pi/l2/BiasAddBiasAddpi/l2/MatMulpi/l2/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
T

pi/l2/ReluRelupi/l2/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-pi/l3/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:

+pi/l3/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
_class
loc:@pi/l3/kernel*
valueB
 *×łÝ˝*
dtype0

+pi/l3/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/l3/kernel*
valueB
 *×łÝ=*
dtype0*
_output_shapes
: 
ç
5pi/l3/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l3/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*
_class
loc:@pi/l3/kernel*
seed2 
Î
+pi/l3/kernel/Initializer/random_uniform/subSub+pi/l3/kernel/Initializer/random_uniform/max+pi/l3/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l3/kernel*
_output_shapes
: 
â
+pi/l3/kernel/Initializer/random_uniform/mulMul5pi/l3/kernel/Initializer/random_uniform/RandomUniform+pi/l3/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:

Ô
'pi/l3/kernel/Initializer/random_uniformAdd+pi/l3/kernel/Initializer/random_uniform/mul+pi/l3/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:

Ľ
pi/l3/kernel
VariableV2*
_class
loc:@pi/l3/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
É
pi/l3/kernel/AssignAssignpi/l3/kernel'pi/l3/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@pi/l3/kernel*
validate_shape(* 
_output_shapes
:

w
pi/l3/kernel/readIdentitypi/l3/kernel*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:


pi/l3/bias/Initializer/zerosConst*
_class
loc:@pi/l3/bias*
valueB*    *
dtype0*
_output_shapes	
:


pi/l3/bias
VariableV2*
shared_name *
_class
loc:@pi/l3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ł
pi/l3/bias/AssignAssign
pi/l3/biaspi/l3/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l3/bias*
validate_shape(*
_output_shapes	
:
l
pi/l3/bias/readIdentity
pi/l3/bias*
_output_shapes	
:*
T0*
_class
loc:@pi/l3/bias

pi/l3/MatMulMatMul
pi/l2/Relupi/l3/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

pi/l3/BiasAddBiasAddpi/l3/MatMulpi/l3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
T

pi/l3/ReluRelupi/l3/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-pi/l4/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/l4/kernel*
valueB"      *
dtype0*
_output_shapes
:

+pi/l4/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/l4/kernel*
valueB
 *   ž*
dtype0*
_output_shapes
: 

+pi/l4/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/l4/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
ç
5pi/l4/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l4/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*
_class
loc:@pi/l4/kernel
Î
+pi/l4/kernel/Initializer/random_uniform/subSub+pi/l4/kernel/Initializer/random_uniform/max+pi/l4/kernel/Initializer/random_uniform/min*
_class
loc:@pi/l4/kernel*
_output_shapes
: *
T0
â
+pi/l4/kernel/Initializer/random_uniform/mulMul5pi/l4/kernel/Initializer/random_uniform/RandomUniform+pi/l4/kernel/Initializer/random_uniform/sub*
_class
loc:@pi/l4/kernel* 
_output_shapes
:
*
T0
Ô
'pi/l4/kernel/Initializer/random_uniformAdd+pi/l4/kernel/Initializer/random_uniform/mul+pi/l4/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l4/kernel* 
_output_shapes
:

Ľ
pi/l4/kernel
VariableV2*
_class
loc:@pi/l4/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
É
pi/l4/kernel/AssignAssignpi/l4/kernel'pi/l4/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@pi/l4/kernel*
validate_shape(* 
_output_shapes
:

w
pi/l4/kernel/readIdentitypi/l4/kernel*
T0*
_class
loc:@pi/l4/kernel* 
_output_shapes
:


pi/l4/bias/Initializer/zerosConst*
_class
loc:@pi/l4/bias*
valueB*    *
dtype0*
_output_shapes	
:


pi/l4/bias
VariableV2*
_class
loc:@pi/l4/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ł
pi/l4/bias/AssignAssign
pi/l4/biaspi/l4/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l4/bias*
validate_shape(*
_output_shapes	
:
l
pi/l4/bias/readIdentity
pi/l4/bias*
_output_shapes	
:*
T0*
_class
loc:@pi/l4/bias

pi/l4/MatMulMatMul
pi/l3/Relupi/l4/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

pi/l4/BiasAddBiasAddpi/l4/MatMulpi/l4/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
T

pi/l4/ReluRelupi/l4/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

,pi/a/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/a/kernel*
valueB"      *
dtype0*
_output_shapes
:

*pi/a/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/a/kernel*
valueB
 *JQZž*
dtype0*
_output_shapes
: 

*pi/a/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/a/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
ă
4pi/a/kernel/Initializer/random_uniform/RandomUniformRandomUniform,pi/a/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	*

seed *
T0*
_class
loc:@pi/a/kernel
Ę
*pi/a/kernel/Initializer/random_uniform/subSub*pi/a/kernel/Initializer/random_uniform/max*pi/a/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/a/kernel*
_output_shapes
: 
Ý
*pi/a/kernel/Initializer/random_uniform/mulMul4pi/a/kernel/Initializer/random_uniform/RandomUniform*pi/a/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*
_class
loc:@pi/a/kernel
Ď
&pi/a/kernel/Initializer/random_uniformAdd*pi/a/kernel/Initializer/random_uniform/mul*pi/a/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/a/kernel*
_output_shapes
:	
Ą
pi/a/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *
_class
loc:@pi/a/kernel*
	container *
shape:	
Ä
pi/a/kernel/AssignAssignpi/a/kernel&pi/a/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@pi/a/kernel*
validate_shape(*
_output_shapes
:	
s
pi/a/kernel/readIdentitypi/a/kernel*
_output_shapes
:	*
T0*
_class
loc:@pi/a/kernel

pi/a/bias/Initializer/zerosConst*
_class
loc:@pi/a/bias*
valueB*    *
dtype0*
_output_shapes
:

	pi/a/bias
VariableV2*
shared_name *
_class
loc:@pi/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ž
pi/a/bias/AssignAssign	pi/a/biaspi/a/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
:
h
pi/a/bias/readIdentity	pi/a/bias*
_output_shapes
:*
T0*
_class
loc:@pi/a/bias

pi/a/MatMulMatMul
pi/l4/Relupi/a/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
}
pi/a/BiasAddBiasAddpi/a/MatMulpi/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
	pi/a/TanhTanhpi/a/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
g
pi/scaled_mu/yConst*%
valueB"Z<?ˇŃ8˝75ˇŃ8*
dtype0*
_output_shapes
:
`
pi/scaled_muMul	pi/a/Tanhpi/scaled_mu/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ľ
0pi/dense/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@pi/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:

.pi/dense/kernel/Initializer/random_uniform/minConst*"
_class
loc:@pi/dense/kernel*
valueB
 *JQZž*
dtype0*
_output_shapes
: 

.pi/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@pi/dense/kernel*
valueB
 *JQZ>*
dtype0
ď
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*"
_class
loc:@pi/dense/kernel*
seed2 
Ú
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@pi/dense/kernel*
_output_shapes
: *
T0
í
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	
ß
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	
Š
pi/dense/kernel
VariableV2*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ô
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*
_output_shapes
:	*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(

pi/dense/kernel/readIdentitypi/dense/kernel*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	*
T0

pi/dense/bias/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0*
_output_shapes
:

pi/dense/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@pi/dense/bias
ž
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:
t
pi/dense/bias/readIdentitypi/dense/bias*
_output_shapes
:*
T0* 
_class
loc:@pi/dense/bias

pi/dense/MatMulMatMul
pi/l4/Relupi/dense/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
pi/dense/SoftplusSoftpluspi/dense/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
pi/scaled_sigma/yConst*%
valueB"Z<?ˇŃ8˝75ˇŃ8*
dtype0*
_output_shapes
:
n
pi/scaled_sigmaMulpi/dense/Softpluspi/scaled_sigma/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Y
pi/Normal/locIdentitypi/scaled_mu*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
pi/Normal/scaleIdentitypi/scaled_sigma*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0oldpi/l1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@oldpi/l1/kernel*
valueB"      

.oldpi/l1/kernel/Initializer/random_uniform/minConst*"
_class
loc:@oldpi/l1/kernel*
valueB
 *°îž*
dtype0*
_output_shapes
: 

.oldpi/l1/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@oldpi/l1/kernel*
valueB
 *°î>*
dtype0*
_output_shapes
: 
ď
8oldpi/l1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l1/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	*

seed *
T0*"
_class
loc:@oldpi/l1/kernel
Ú
.oldpi/l1/kernel/Initializer/random_uniform/subSub.oldpi/l1/kernel/Initializer/random_uniform/max.oldpi/l1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l1/kernel*
_output_shapes
: 
í
.oldpi/l1/kernel/Initializer/random_uniform/mulMul8oldpi/l1/kernel/Initializer/random_uniform/RandomUniform.oldpi/l1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@oldpi/l1/kernel*
_output_shapes
:	
ß
*oldpi/l1/kernel/Initializer/random_uniformAdd.oldpi/l1/kernel/Initializer/random_uniform/mul.oldpi/l1/kernel/Initializer/random_uniform/min*
_output_shapes
:	*
T0*"
_class
loc:@oldpi/l1/kernel
Š
oldpi/l1/kernel
VariableV2*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name *"
_class
loc:@oldpi/l1/kernel
Ô
oldpi/l1/kernel/AssignAssignoldpi/l1/kernel*oldpi/l1/kernel/Initializer/random_uniform*"
_class
loc:@oldpi/l1/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0

oldpi/l1/kernel/readIdentityoldpi/l1/kernel*
T0*"
_class
loc:@oldpi/l1/kernel*
_output_shapes
:	

oldpi/l1/bias/Initializer/zerosConst* 
_class
loc:@oldpi/l1/bias*
valueB*    *
dtype0*
_output_shapes	
:

oldpi/l1/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name * 
_class
loc:@oldpi/l1/bias
ż
oldpi/l1/bias/AssignAssignoldpi/l1/biasoldpi/l1/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@oldpi/l1/bias*
validate_shape(*
_output_shapes	
:
u
oldpi/l1/bias/readIdentityoldpi/l1/bias*
T0* 
_class
loc:@oldpi/l1/bias*
_output_shapes	
:

oldpi/l1/MatMulMatMulstateoldpi/l1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

oldpi/l1/BiasAddBiasAddoldpi/l1/MatMuloldpi/l1/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Z
oldpi/l1/ReluReluoldpi/l1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0oldpi/l2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*"
_class
loc:@oldpi/l2/kernel*
valueB"      *
dtype0

.oldpi/l2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@oldpi/l2/kernel*
valueB
 *×łÝ˝

.oldpi/l2/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@oldpi/l2/kernel*
valueB
 *×łÝ=*
dtype0*
_output_shapes
: 
đ
8oldpi/l2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l2/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*"
_class
loc:@oldpi/l2/kernel*
seed2 
Ú
.oldpi/l2/kernel/Initializer/random_uniform/subSub.oldpi/l2/kernel/Initializer/random_uniform/max.oldpi/l2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l2/kernel*
_output_shapes
: 
î
.oldpi/l2/kernel/Initializer/random_uniform/mulMul8oldpi/l2/kernel/Initializer/random_uniform/RandomUniform.oldpi/l2/kernel/Initializer/random_uniform/sub*"
_class
loc:@oldpi/l2/kernel* 
_output_shapes
:
*
T0
ŕ
*oldpi/l2/kernel/Initializer/random_uniformAdd.oldpi/l2/kernel/Initializer/random_uniform/mul.oldpi/l2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l2/kernel* 
_output_shapes
:

Ť
oldpi/l2/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *"
_class
loc:@oldpi/l2/kernel*
	container *
shape:

Ő
oldpi/l2/kernel/AssignAssignoldpi/l2/kernel*oldpi/l2/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*"
_class
loc:@oldpi/l2/kernel

oldpi/l2/kernel/readIdentityoldpi/l2/kernel*"
_class
loc:@oldpi/l2/kernel* 
_output_shapes
:
*
T0

oldpi/l2/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:* 
_class
loc:@oldpi/l2/bias*
valueB*    

oldpi/l2/bias
VariableV2*
shared_name * 
_class
loc:@oldpi/l2/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ż
oldpi/l2/bias/AssignAssignoldpi/l2/biasoldpi/l2/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@oldpi/l2/bias*
validate_shape(
u
oldpi/l2/bias/readIdentityoldpi/l2/bias*
T0* 
_class
loc:@oldpi/l2/bias*
_output_shapes	
:

oldpi/l2/MatMulMatMuloldpi/l1/Reluoldpi/l2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

oldpi/l2/BiasAddBiasAddoldpi/l2/MatMuloldpi/l2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
oldpi/l2/ReluReluoldpi/l2/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0oldpi/l3/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@oldpi/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:

.oldpi/l3/kernel/Initializer/random_uniform/minConst*"
_class
loc:@oldpi/l3/kernel*
valueB
 *×łÝ˝*
dtype0*
_output_shapes
: 

.oldpi/l3/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@oldpi/l3/kernel*
valueB
 *×łÝ=*
dtype0*
_output_shapes
: 
đ
8oldpi/l3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l3/kernel/Initializer/random_uniform/shape*"
_class
loc:@oldpi/l3/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0
Ú
.oldpi/l3/kernel/Initializer/random_uniform/subSub.oldpi/l3/kernel/Initializer/random_uniform/max.oldpi/l3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l3/kernel*
_output_shapes
: 
î
.oldpi/l3/kernel/Initializer/random_uniform/mulMul8oldpi/l3/kernel/Initializer/random_uniform/RandomUniform.oldpi/l3/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@oldpi/l3/kernel* 
_output_shapes
:

ŕ
*oldpi/l3/kernel/Initializer/random_uniformAdd.oldpi/l3/kernel/Initializer/random_uniform/mul.oldpi/l3/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*"
_class
loc:@oldpi/l3/kernel
Ť
oldpi/l3/kernel
VariableV2*"
_class
loc:@oldpi/l3/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Ő
oldpi/l3/kernel/AssignAssignoldpi/l3/kernel*oldpi/l3/kernel/Initializer/random_uniform*"
_class
loc:@oldpi/l3/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

oldpi/l3/kernel/readIdentityoldpi/l3/kernel*
T0*"
_class
loc:@oldpi/l3/kernel* 
_output_shapes
:


oldpi/l3/bias/Initializer/zerosConst* 
_class
loc:@oldpi/l3/bias*
valueB*    *
dtype0*
_output_shapes	
:

oldpi/l3/bias
VariableV2* 
_class
loc:@oldpi/l3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ż
oldpi/l3/bias/AssignAssignoldpi/l3/biasoldpi/l3/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@oldpi/l3/bias*
validate_shape(*
_output_shapes	
:
u
oldpi/l3/bias/readIdentityoldpi/l3/bias*
T0* 
_class
loc:@oldpi/l3/bias*
_output_shapes	
:

oldpi/l3/MatMulMatMuloldpi/l2/Reluoldpi/l3/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

oldpi/l3/BiasAddBiasAddoldpi/l3/MatMuloldpi/l3/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Z
oldpi/l3/ReluReluoldpi/l3/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ľ
0oldpi/l4/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@oldpi/l4/kernel*
valueB"      *
dtype0*
_output_shapes
:

.oldpi/l4/kernel/Initializer/random_uniform/minConst*"
_class
loc:@oldpi/l4/kernel*
valueB
 *   ž*
dtype0*
_output_shapes
: 

.oldpi/l4/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@oldpi/l4/kernel*
valueB
 *   >*
dtype0
đ
8oldpi/l4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l4/kernel/Initializer/random_uniform/shape*"
_class
loc:@oldpi/l4/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0
Ú
.oldpi/l4/kernel/Initializer/random_uniform/subSub.oldpi/l4/kernel/Initializer/random_uniform/max.oldpi/l4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l4/kernel*
_output_shapes
: 
î
.oldpi/l4/kernel/Initializer/random_uniform/mulMul8oldpi/l4/kernel/Initializer/random_uniform/RandomUniform.oldpi/l4/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@oldpi/l4/kernel* 
_output_shapes
:

ŕ
*oldpi/l4/kernel/Initializer/random_uniformAdd.oldpi/l4/kernel/Initializer/random_uniform/mul.oldpi/l4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l4/kernel* 
_output_shapes
:

Ť
oldpi/l4/kernel
VariableV2*"
_class
loc:@oldpi/l4/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Ő
oldpi/l4/kernel/AssignAssignoldpi/l4/kernel*oldpi/l4/kernel/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*"
_class
loc:@oldpi/l4/kernel*
validate_shape(

oldpi/l4/kernel/readIdentityoldpi/l4/kernel*"
_class
loc:@oldpi/l4/kernel* 
_output_shapes
:
*
T0

oldpi/l4/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:* 
_class
loc:@oldpi/l4/bias*
valueB*    

oldpi/l4/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name * 
_class
loc:@oldpi/l4/bias*
	container 
ż
oldpi/l4/bias/AssignAssignoldpi/l4/biasoldpi/l4/bias/Initializer/zeros*
T0* 
_class
loc:@oldpi/l4/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
u
oldpi/l4/bias/readIdentityoldpi/l4/bias*
T0* 
_class
loc:@oldpi/l4/bias*
_output_shapes	
:

oldpi/l4/MatMulMatMuloldpi/l3/Reluoldpi/l4/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

oldpi/l4/BiasAddBiasAddoldpi/l4/MatMuloldpi/l4/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Z
oldpi/l4/ReluReluoldpi/l4/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
/oldpi/a/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@oldpi/a/kernel*
valueB"      *
dtype0*
_output_shapes
:

-oldpi/a/kernel/Initializer/random_uniform/minConst*!
_class
loc:@oldpi/a/kernel*
valueB
 *JQZž*
dtype0*
_output_shapes
: 

-oldpi/a/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *!
_class
loc:@oldpi/a/kernel*
valueB
 *JQZ>
ě
7oldpi/a/kernel/Initializer/random_uniform/RandomUniformRandomUniform/oldpi/a/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@oldpi/a/kernel*
seed2 *
dtype0*
_output_shapes
:	*

seed 
Ö
-oldpi/a/kernel/Initializer/random_uniform/subSub-oldpi/a/kernel/Initializer/random_uniform/max-oldpi/a/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@oldpi/a/kernel
é
-oldpi/a/kernel/Initializer/random_uniform/mulMul7oldpi/a/kernel/Initializer/random_uniform/RandomUniform-oldpi/a/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@oldpi/a/kernel*
_output_shapes
:	
Ű
)oldpi/a/kernel/Initializer/random_uniformAdd-oldpi/a/kernel/Initializer/random_uniform/mul-oldpi/a/kernel/Initializer/random_uniform/min*!
_class
loc:@oldpi/a/kernel*
_output_shapes
:	*
T0
§
oldpi/a/kernel
VariableV2*
shared_name *!
_class
loc:@oldpi/a/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
Đ
oldpi/a/kernel/AssignAssignoldpi/a/kernel)oldpi/a/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*!
_class
loc:@oldpi/a/kernel
|
oldpi/a/kernel/readIdentityoldpi/a/kernel*
T0*!
_class
loc:@oldpi/a/kernel*
_output_shapes
:	

oldpi/a/bias/Initializer/zerosConst*
_output_shapes
:*
_class
loc:@oldpi/a/bias*
valueB*    *
dtype0

oldpi/a/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@oldpi/a/bias*
	container 
ş
oldpi/a/bias/AssignAssignoldpi/a/biasoldpi/a/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@oldpi/a/bias*
validate_shape(*
_output_shapes
:
q
oldpi/a/bias/readIdentityoldpi/a/bias*
_class
loc:@oldpi/a/bias*
_output_shapes
:*
T0

oldpi/a/MatMulMatMuloldpi/l4/Reluoldpi/a/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

oldpi/a/BiasAddBiasAddoldpi/a/MatMuloldpi/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
oldpi/a/TanhTanholdpi/a/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
oldpi/scaled_mu/yConst*%
valueB"Z<?ˇŃ8˝75ˇŃ8*
dtype0*
_output_shapes
:
i
oldpi/scaled_muMuloldpi/a/Tanholdpi/scaled_mu/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ť
3oldpi/dense/kernel/Initializer/random_uniform/shapeConst*%
_class
loc:@oldpi/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:

1oldpi/dense/kernel/Initializer/random_uniform/minConst*%
_class
loc:@oldpi/dense/kernel*
valueB
 *JQZž*
dtype0*
_output_shapes
: 

1oldpi/dense/kernel/Initializer/random_uniform/maxConst*%
_class
loc:@oldpi/dense/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
ř
;oldpi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform3oldpi/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*%
_class
loc:@oldpi/dense/kernel*
seed2 
ć
1oldpi/dense/kernel/Initializer/random_uniform/subSub1oldpi/dense/kernel/Initializer/random_uniform/max1oldpi/dense/kernel/Initializer/random_uniform/min*
T0*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
: 
ů
1oldpi/dense/kernel/Initializer/random_uniform/mulMul;oldpi/dense/kernel/Initializer/random_uniform/RandomUniform1oldpi/dense/kernel/Initializer/random_uniform/sub*
T0*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
:	
ë
-oldpi/dense/kernel/Initializer/random_uniformAdd1oldpi/dense/kernel/Initializer/random_uniform/mul1oldpi/dense/kernel/Initializer/random_uniform/min*
T0*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
:	
Ż
oldpi/dense/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *%
_class
loc:@oldpi/dense/kernel*
	container *
shape:	
ŕ
oldpi/dense/kernel/AssignAssignoldpi/dense/kernel-oldpi/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*%
_class
loc:@oldpi/dense/kernel*
validate_shape(*
_output_shapes
:	

oldpi/dense/kernel/readIdentityoldpi/dense/kernel*
T0*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
:	

"oldpi/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*#
_class
loc:@oldpi/dense/bias*
valueB*    
Ą
oldpi/dense/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *#
_class
loc:@oldpi/dense/bias*
	container *
shape:
Ę
oldpi/dense/bias/AssignAssignoldpi/dense/bias"oldpi/dense/bias/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*#
_class
loc:@oldpi/dense/bias*
validate_shape(
}
oldpi/dense/bias/readIdentityoldpi/dense/bias*
_output_shapes
:*
T0*#
_class
loc:@oldpi/dense/bias

oldpi/dense/MatMulMatMuloldpi/l4/Reluoldpi/dense/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

oldpi/dense/BiasAddBiasAddoldpi/dense/MatMuloldpi/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
oldpi/dense/SoftplusSoftplusoldpi/dense/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
m
oldpi/scaled_sigma/yConst*
dtype0*
_output_shapes
:*%
valueB"Z<?ˇŃ8˝75ˇŃ8
w
oldpi/scaled_sigmaMuloldpi/dense/Softplusoldpi/scaled_sigma/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
_
oldpi/Normal/locIdentityoldpi/scaled_mu*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
oldpi/Normal/scaleIdentityoldpi/scaled_sigma*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
pi/Normal/sample/sample_shapeConst*
value	B :*
dtype0*
_output_shapes
: 
i
pi/Normal/sample/sample_shape_1Const*
_output_shapes
:*
valueB:*
dtype0
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
Ş
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
É
pi/Normal/sample/concatConcatV2 pi/Normal/sample/concat/values_0*pi/Normal/batch_shape_tensor/BroadcastArgspi/Normal/sample/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
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
 *  ?*
dtype0*
_output_shapes
: 
É
3pi/Normal/sample/random_normal/RandomStandardNormalRandomStandardNormalpi/Normal/sample/concat*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
seed2 *

seed *
T0*
dtype0
Ä
"pi/Normal/sample/random_normal/mulMul3pi/Normal/sample/random_normal/RandomStandardNormal%pi/Normal/sample/random_normal/stddev*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
­
pi/Normal/sample/random_normalAdd"pi/Normal/sample/random_normal/mul#pi/Normal/sample/random_normal/mean*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

pi/Normal/sample/mulMulpi/Normal/sample/random_normalpi/Normal/scale*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
pi/Normal/sample/addAddpi/Normal/sample/mulpi/Normal/loc*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
pi/Normal/sample/ShapeShapepi/Normal/sample/add*
_output_shapes
:*
T0*
out_type0
n
$pi/Normal/sample/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
p
&pi/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
p
&pi/Normal/sample/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ň
pi/Normal/sample/strided_sliceStridedSlicepi/Normal/sample/Shape$pi/Normal/sample/strided_slice/stack&pi/Normal/sample/strided_slice/stack_1&pi/Normal/sample/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
T0*
Index0*
shrink_axis_mask 
`
pi/Normal/sample/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ŕ
pi/Normal/sample/concat_1ConcatV2pi/Normal/sample/sample_shape_1pi/Normal/sample/strided_slicepi/Normal/sample/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N

pi/Normal/sample/ReshapeReshapepi/Normal/sample/addpi/Normal/sample/concat_1*
T0*
Tshape0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

sample_action/SqueezeSqueezepi/Normal/sample/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
 
¸
update_oldpi/AssignAssignoldpi/l1/kernelpi/l1/kernel/read*
validate_shape(*
_output_shapes
:	*
use_locking( *
T0*"
_class
loc:@oldpi/l1/kernel
°
update_oldpi/Assign_1Assignoldpi/l1/biaspi/l1/bias/read* 
_class
loc:@oldpi/l1/bias*
validate_shape(*
_output_shapes	
:*
use_locking( *
T0
ť
update_oldpi/Assign_2Assignoldpi/l2/kernelpi/l2/kernel/read* 
_output_shapes
:
*
use_locking( *
T0*"
_class
loc:@oldpi/l2/kernel*
validate_shape(
°
update_oldpi/Assign_3Assignoldpi/l2/biaspi/l2/bias/read*
_output_shapes	
:*
use_locking( *
T0* 
_class
loc:@oldpi/l2/bias*
validate_shape(
ť
update_oldpi/Assign_4Assignoldpi/l3/kernelpi/l3/kernel/read*
use_locking( *
T0*"
_class
loc:@oldpi/l3/kernel*
validate_shape(* 
_output_shapes
:

°
update_oldpi/Assign_5Assignoldpi/l3/biaspi/l3/bias/read* 
_class
loc:@oldpi/l3/bias*
validate_shape(*
_output_shapes	
:*
use_locking( *
T0
ť
update_oldpi/Assign_6Assignoldpi/l4/kernelpi/l4/kernel/read*"
_class
loc:@oldpi/l4/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking( *
T0
°
update_oldpi/Assign_7Assignoldpi/l4/biaspi/l4/bias/read* 
_class
loc:@oldpi/l4/bias*
validate_shape(*
_output_shapes	
:*
use_locking( *
T0
ˇ
update_oldpi/Assign_8Assignoldpi/a/kernelpi/a/kernel/read*
validate_shape(*
_output_shapes
:	*
use_locking( *
T0*!
_class
loc:@oldpi/a/kernel
Ź
update_oldpi/Assign_9Assignoldpi/a/biaspi/a/bias/read*
_output_shapes
:*
use_locking( *
T0*
_class
loc:@oldpi/a/bias*
validate_shape(
Ä
update_oldpi/Assign_10Assignoldpi/dense/kernelpi/dense/kernel/read*%
_class
loc:@oldpi/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking( *
T0
š
update_oldpi/Assign_11Assignoldpi/dense/biaspi/dense/bias/read*
use_locking( *
T0*#
_class
loc:@oldpi/dense/bias*
validate_shape(*
_output_shapes
:
i
actionPlaceholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
l
	advantagePlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
pi/Normal/prob/standardize/subSubactionpi/Normal/loc*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"pi/Normal/prob/standardize/truedivRealDivpi/Normal/prob/standardize/subpi/Normal/scale*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
pi/Normal/prob/SquareSquare"pi/Normal/prob/standardize/truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Y
pi/Normal/prob/mul/xConst*
valueB
 *   ż*
dtype0*
_output_shapes
: 
x
pi/Normal/prob/mulMulpi/Normal/prob/mul/xpi/Normal/prob/Square*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
pi/Normal/prob/LogLogpi/Normal/scale*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
pi/Normal/prob/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *?k?
u
pi/Normal/prob/addAddpi/Normal/prob/add/xpi/Normal/prob/Log*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
pi/Normal/prob/subSubpi/Normal/prob/mulpi/Normal/prob/add*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
_
pi/Normal/prob/ExpExppi/Normal/prob/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
!oldpi/Normal/prob/standardize/subSubactionoldpi/Normal/loc*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

%oldpi/Normal/prob/standardize/truedivRealDiv!oldpi/Normal/prob/standardize/suboldpi/Normal/scale*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
{
oldpi/Normal/prob/SquareSquare%oldpi/Normal/prob/standardize/truediv*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
oldpi/Normal/prob/mul/xConst*
valueB
 *   ż*
dtype0*
_output_shapes
: 

oldpi/Normal/prob/mulMuloldpi/Normal/prob/mul/xoldpi/Normal/prob/Square*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
oldpi/Normal/prob/LogLogoldpi/Normal/scale*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
oldpi/Normal/prob/add/xConst*
valueB
 *?k?*
dtype0*
_output_shapes
: 
~
oldpi/Normal/prob/addAddoldpi/Normal/prob/add/xoldpi/Normal/prob/Log*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
|
oldpi/Normal/prob/subSuboldpi/Normal/prob/muloldpi/Normal/prob/add*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
e
oldpi/Normal/prob/ExpExpoldpi/Normal/prob/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
~
loss/surrogate/truedivRealDivpi/Normal/prob/Expoldpi/Normal/prob/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
n
loss/surrogate/mulMulloss/surrogate/truediv	advantage*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
loss/clip_by_value/Minimum/yConst*
valueB
 *?*
dtype0*
_output_shapes
: 

loss/clip_by_value/MinimumMinimumloss/surrogate/truedivloss/clip_by_value/Minimum/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
loss/clip_by_value/yConst*
valueB
 *ÍĚL?*
dtype0*
_output_shapes
: 

loss/clip_by_valueMaximumloss/clip_by_value/Minimumloss/clip_by_value/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
loss/mulMulloss/clip_by_value	advantage*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
g
loss/MinimumMinimumloss/surrogate/mulloss/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
[

loss/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
i
	loss/MeanMeanloss/Minimum
loss/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
;
loss/NegNeg	loss/Mean*
T0*
_output_shapes
: 
Y
atrain/gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
_
atrain/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

atrain/gradients/FillFillatrain/gradients/Shapeatrain/gradients/grad_ys_0*

index_type0*
_output_shapes
: *
T0
a
"atrain/gradients/loss/Neg_grad/NegNegatrain/gradients/Fill*
T0*
_output_shapes
: 
~
-atrain/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ź
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
Ŕ
$atrain/gradients/loss/Mean_grad/TileTile'atrain/gradients/loss/Mean_grad/Reshape%atrain/gradients/loss/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
ş
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
ž
&atrain/gradients/loss/Mean_grad/Prod_1Prod'atrain/gradients/loss/Mean_grad/Shape_2'atrain/gradients/loss/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
k
)atrain/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ś
'atrain/gradients/loss/Mean_grad/MaximumMaximum&atrain/gradients/loss/Mean_grad/Prod_1)atrain/gradients/loss/Mean_grad/Maximum/y*
_output_shapes
: *
T0
¤
(atrain/gradients/loss/Mean_grad/floordivFloorDiv$atrain/gradients/loss/Mean_grad/Prod'atrain/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 

$atrain/gradients/loss/Mean_grad/CastCast(atrain/gradients/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
°
'atrain/gradients/loss/Mean_grad/truedivRealDiv$atrain/gradients/loss/Mean_grad/Tile$atrain/gradients/loss/Mean_grad/Cast*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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

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
Đ
(atrain/gradients/loss/Minimum_grad/zerosFill*atrain/gradients/loss/Minimum_grad/Shape_2.atrain/gradients/loss/Minimum_grad/zeros/Const*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

,atrain/gradients/loss/Minimum_grad/LessEqual	LessEqualloss/surrogate/mulloss/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
8atrain/gradients/loss/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs(atrain/gradients/loss/Minimum_grad/Shape*atrain/gradients/loss/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ć
)atrain/gradients/loss/Minimum_grad/SelectSelect,atrain/gradients/loss/Minimum_grad/LessEqual'atrain/gradients/loss/Mean_grad/truediv(atrain/gradients/loss/Minimum_grad/zeros*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
+atrain/gradients/loss/Minimum_grad/Select_1Select,atrain/gradients/loss/Minimum_grad/LessEqual(atrain/gradients/loss/Minimum_grad/zeros'atrain/gradients/loss/Mean_grad/truediv*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
&atrain/gradients/loss/Minimum_grad/SumSum)atrain/gradients/loss/Minimum_grad/Select8atrain/gradients/loss/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ç
*atrain/gradients/loss/Minimum_grad/ReshapeReshape&atrain/gradients/loss/Minimum_grad/Sum(atrain/gradients/loss/Minimum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
(atrain/gradients/loss/Minimum_grad/Sum_1Sum+atrain/gradients/loss/Minimum_grad/Select_1:atrain/gradients/loss/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Í
,atrain/gradients/loss/Minimum_grad/Reshape_1Reshape(atrain/gradients/loss/Minimum_grad/Sum_1*atrain/gradients/loss/Minimum_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

3atrain/gradients/loss/Minimum_grad/tuple/group_depsNoOp+^atrain/gradients/loss/Minimum_grad/Reshape-^atrain/gradients/loss/Minimum_grad/Reshape_1

;atrain/gradients/loss/Minimum_grad/tuple/control_dependencyIdentity*atrain/gradients/loss/Minimum_grad/Reshape4^atrain/gradients/loss/Minimum_grad/tuple/group_deps*
T0*=
_class3
1/loc:@atrain/gradients/loss/Minimum_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
=atrain/gradients/loss/Minimum_grad/tuple/control_dependency_1Identity,atrain/gradients/loss/Minimum_grad/Reshape_14^atrain/gradients/loss/Minimum_grad/tuple/group_deps*
T0*?
_class5
31loc:@atrain/gradients/loss/Minimum_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

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
ö
>atrain/gradients/loss/surrogate/mul_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/loss/surrogate/mul_grad/Shape0atrain/gradients/loss/surrogate/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
­
,atrain/gradients/loss/surrogate/mul_grad/MulMul;atrain/gradients/loss/Minimum_grad/tuple/control_dependency	advantage*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
,atrain/gradients/loss/surrogate/mul_grad/SumSum,atrain/gradients/loss/surrogate/mul_grad/Mul>atrain/gradients/loss/surrogate/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ů
0atrain/gradients/loss/surrogate/mul_grad/ReshapeReshape,atrain/gradients/loss/surrogate/mul_grad/Sum.atrain/gradients/loss/surrogate/mul_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ź
.atrain/gradients/loss/surrogate/mul_grad/Mul_1Mulloss/surrogate/truediv;atrain/gradients/loss/Minimum_grad/tuple/control_dependency*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ç
.atrain/gradients/loss/surrogate/mul_grad/Sum_1Sum.atrain/gradients/loss/surrogate/mul_grad/Mul_1@atrain/gradients/loss/surrogate/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ß
2atrain/gradients/loss/surrogate/mul_grad/Reshape_1Reshape.atrain/gradients/loss/surrogate/mul_grad/Sum_10atrain/gradients/loss/surrogate/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
9atrain/gradients/loss/surrogate/mul_grad/tuple/group_depsNoOp1^atrain/gradients/loss/surrogate/mul_grad/Reshape3^atrain/gradients/loss/surrogate/mul_grad/Reshape_1
˛
Aatrain/gradients/loss/surrogate/mul_grad/tuple/control_dependencyIdentity0atrain/gradients/loss/surrogate/mul_grad/Reshape:^atrain/gradients/loss/surrogate/mul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*C
_class9
75loc:@atrain/gradients/loss/surrogate/mul_grad/Reshape
¸
Catrain/gradients/loss/surrogate/mul_grad/tuple/control_dependency_1Identity2atrain/gradients/loss/surrogate/mul_grad/Reshape_1:^atrain/gradients/loss/surrogate/mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/loss/surrogate/mul_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
$atrain/gradients/loss/mul_grad/ShapeShapeloss/clip_by_value*
T0*
out_type0*
_output_shapes
:
o
&atrain/gradients/loss/mul_grad/Shape_1Shape	advantage*
T0*
out_type0*
_output_shapes
:
Ř
4atrain/gradients/loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs$atrain/gradients/loss/mul_grad/Shape&atrain/gradients/loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
"atrain/gradients/loss/mul_grad/MulMul=atrain/gradients/loss/Minimum_grad/tuple/control_dependency_1	advantage*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
"atrain/gradients/loss/mul_grad/SumSum"atrain/gradients/loss/mul_grad/Mul4atrain/gradients/loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ť
&atrain/gradients/loss/mul_grad/ReshapeReshape"atrain/gradients/loss/mul_grad/Sum$atrain/gradients/loss/mul_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
°
$atrain/gradients/loss/mul_grad/Mul_1Mulloss/clip_by_value=atrain/gradients/loss/Minimum_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
$atrain/gradients/loss/mul_grad/Sum_1Sum$atrain/gradients/loss/mul_grad/Mul_16atrain/gradients/loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Á
(atrain/gradients/loss/mul_grad/Reshape_1Reshape$atrain/gradients/loss/mul_grad/Sum_1&atrain/gradients/loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

/atrain/gradients/loss/mul_grad/tuple/group_depsNoOp'^atrain/gradients/loss/mul_grad/Reshape)^atrain/gradients/loss/mul_grad/Reshape_1

7atrain/gradients/loss/mul_grad/tuple/control_dependencyIdentity&atrain/gradients/loss/mul_grad/Reshape0^atrain/gradients/loss/mul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@atrain/gradients/loss/mul_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

9atrain/gradients/loss/mul_grad/tuple/control_dependency_1Identity(atrain/gradients/loss/mul_grad/Reshape_10^atrain/gradients/loss/mul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*;
_class1
/-loc:@atrain/gradients/loss/mul_grad/Reshape_1

.atrain/gradients/loss/clip_by_value_grad/ShapeShapeloss/clip_by_value/Minimum*
T0*
out_type0*
_output_shapes
:
s
0atrain/gradients/loss/clip_by_value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
§
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
â
.atrain/gradients/loss/clip_by_value_grad/zerosFill0atrain/gradients/loss/clip_by_value_grad/Shape_24atrain/gradients/loss/clip_by_value_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
5atrain/gradients/loss/clip_by_value_grad/GreaterEqualGreaterEqualloss/clip_by_value/Minimumloss/clip_by_value/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ö
>atrain/gradients/loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/loss/clip_by_value_grad/Shape0atrain/gradients/loss/clip_by_value_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

/atrain/gradients/loss/clip_by_value_grad/SelectSelect5atrain/gradients/loss/clip_by_value_grad/GreaterEqual7atrain/gradients/loss/mul_grad/tuple/control_dependency.atrain/gradients/loss/clip_by_value_grad/zeros*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

1atrain/gradients/loss/clip_by_value_grad/Select_1Select5atrain/gradients/loss/clip_by_value_grad/GreaterEqual.atrain/gradients/loss/clip_by_value_grad/zeros7atrain/gradients/loss/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
,atrain/gradients/loss/clip_by_value_grad/SumSum/atrain/gradients/loss/clip_by_value_grad/Select>atrain/gradients/loss/clip_by_value_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ů
0atrain/gradients/loss/clip_by_value_grad/ReshapeReshape,atrain/gradients/loss/clip_by_value_grad/Sum.atrain/gradients/loss/clip_by_value_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ę
.atrain/gradients/loss/clip_by_value_grad/Sum_1Sum1atrain/gradients/loss/clip_by_value_grad/Select_1@atrain/gradients/loss/clip_by_value_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Î
2atrain/gradients/loss/clip_by_value_grad/Reshape_1Reshape.atrain/gradients/loss/clip_by_value_grad/Sum_10atrain/gradients/loss/clip_by_value_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
Š
9atrain/gradients/loss/clip_by_value_grad/tuple/group_depsNoOp1^atrain/gradients/loss/clip_by_value_grad/Reshape3^atrain/gradients/loss/clip_by_value_grad/Reshape_1
˛
Aatrain/gradients/loss/clip_by_value_grad/tuple/control_dependencyIdentity0atrain/gradients/loss/clip_by_value_grad/Reshape:^atrain/gradients/loss/clip_by_value_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*C
_class9
75loc:@atrain/gradients/loss/clip_by_value_grad/Reshape
§
Catrain/gradients/loss/clip_by_value_grad/tuple/control_dependency_1Identity2atrain/gradients/loss/clip_by_value_grad/Reshape_1:^atrain/gradients/loss/clip_by_value_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/loss/clip_by_value_grad/Reshape_1*
_output_shapes
: 

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
š
8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_2ShapeAatrain/gradients/loss/clip_by_value_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0

<atrain/gradients/loss/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ú
6atrain/gradients/loss/clip_by_value/Minimum_grad/zerosFill8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_2<atrain/gradients/loss/clip_by_value/Minimum_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
:atrain/gradients/loss/clip_by_value/Minimum_grad/LessEqual	LessEqualloss/surrogate/truedivloss/clip_by_value/Minimum/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Fatrain/gradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs6atrain/gradients/loss/clip_by_value/Minimum_grad/Shape8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ş
7atrain/gradients/loss/clip_by_value/Minimum_grad/SelectSelect:atrain/gradients/loss/clip_by_value/Minimum_grad/LessEqualAatrain/gradients/loss/clip_by_value_grad/tuple/control_dependency6atrain/gradients/loss/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
9atrain/gradients/loss/clip_by_value/Minimum_grad/Select_1Select:atrain/gradients/loss/clip_by_value/Minimum_grad/LessEqual6atrain/gradients/loss/clip_by_value/Minimum_grad/zerosAatrain/gradients/loss/clip_by_value_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
4atrain/gradients/loss/clip_by_value/Minimum_grad/SumSum7atrain/gradients/loss/clip_by_value/Minimum_grad/SelectFatrain/gradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ń
8atrain/gradients/loss/clip_by_value/Minimum_grad/ReshapeReshape4atrain/gradients/loss/clip_by_value/Minimum_grad/Sum6atrain/gradients/loss/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

6atrain/gradients/loss/clip_by_value/Minimum_grad/Sum_1Sum9atrain/gradients/loss/clip_by_value/Minimum_grad/Select_1Hatrain/gradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ć
:atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1Reshape6atrain/gradients/loss/clip_by_value/Minimum_grad/Sum_18atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Á
Aatrain/gradients/loss/clip_by_value/Minimum_grad/tuple/group_depsNoOp9^atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape;^atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1
Ň
Iatrain/gradients/loss/clip_by_value/Minimum_grad/tuple/control_dependencyIdentity8atrain/gradients/loss/clip_by_value/Minimum_grad/ReshapeB^atrain/gradients/loss/clip_by_value/Minimum_grad/tuple/group_deps*
T0*K
_classA
?=loc:@atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
Katrain/gradients/loss/clip_by_value/Minimum_grad/tuple/control_dependency_1Identity:atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1B^atrain/gradients/loss/clip_by_value/Minimum_grad/tuple/group_deps*
T0*M
_classC
A?loc:@atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1*
_output_shapes
: 
Ť
atrain/gradients/AddNAddNAatrain/gradients/loss/surrogate/mul_grad/tuple/control_dependencyIatrain/gradients/loss/clip_by_value/Minimum_grad/tuple/control_dependency*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*C
_class9
75loc:@atrain/gradients/loss/surrogate/mul_grad/Reshape

2atrain/gradients/loss/surrogate/truediv_grad/ShapeShapepi/Normal/prob/Exp*
T0*
out_type0*
_output_shapes
:

4atrain/gradients/loss/surrogate/truediv_grad/Shape_1Shapeoldpi/Normal/prob/Exp*
T0*
out_type0*
_output_shapes
:

Batrain/gradients/loss/surrogate/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs2atrain/gradients/loss/surrogate/truediv_grad/Shape4atrain/gradients/loss/surrogate/truediv_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

4atrain/gradients/loss/surrogate/truediv_grad/RealDivRealDivatrain/gradients/AddNoldpi/Normal/prob/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ń
0atrain/gradients/loss/surrogate/truediv_grad/SumSum4atrain/gradients/loss/surrogate/truediv_grad/RealDivBatrain/gradients/loss/surrogate/truediv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ĺ
4atrain/gradients/loss/surrogate/truediv_grad/ReshapeReshape0atrain/gradients/loss/surrogate/truediv_grad/Sum2atrain/gradients/loss/surrogate/truediv_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
}
0atrain/gradients/loss/surrogate/truediv_grad/NegNegpi/Normal/prob/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ź
6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_1RealDiv0atrain/gradients/loss/surrogate/truediv_grad/Negoldpi/Normal/prob/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_2RealDiv6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_1oldpi/Normal/prob/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
0atrain/gradients/loss/surrogate/truediv_grad/mulMulatrain/gradients/AddN6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ń
2atrain/gradients/loss/surrogate/truediv_grad/Sum_1Sum0atrain/gradients/loss/surrogate/truediv_grad/mulDatrain/gradients/loss/surrogate/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ë
6atrain/gradients/loss/surrogate/truediv_grad/Reshape_1Reshape2atrain/gradients/loss/surrogate/truediv_grad/Sum_14atrain/gradients/loss/surrogate/truediv_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ľ
=atrain/gradients/loss/surrogate/truediv_grad/tuple/group_depsNoOp5^atrain/gradients/loss/surrogate/truediv_grad/Reshape7^atrain/gradients/loss/surrogate/truediv_grad/Reshape_1
Â
Eatrain/gradients/loss/surrogate/truediv_grad/tuple/control_dependencyIdentity4atrain/gradients/loss/surrogate/truediv_grad/Reshape>^atrain/gradients/loss/surrogate/truediv_grad/tuple/group_deps*
T0*G
_class=
;9loc:@atrain/gradients/loss/surrogate/truediv_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
Gatrain/gradients/loss/surrogate/truediv_grad/tuple/control_dependency_1Identity6atrain/gradients/loss/surrogate/truediv_grad/Reshape_1>^atrain/gradients/loss/surrogate/truediv_grad/tuple/group_deps*I
_class?
=;loc:@atrain/gradients/loss/surrogate/truediv_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ŕ
,atrain/gradients/pi/Normal/prob/Exp_grad/mulMulEatrain/gradients/loss/surrogate/truediv_grad/tuple/control_dependencypi/Normal/prob/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

.atrain/gradients/pi/Normal/prob/sub_grad/ShapeShapepi/Normal/prob/mul*
T0*
out_type0*
_output_shapes
:

0atrain/gradients/pi/Normal/prob/sub_grad/Shape_1Shapepi/Normal/prob/add*
T0*
out_type0*
_output_shapes
:
ö
>atrain/gradients/pi/Normal/prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/pi/Normal/prob/sub_grad/Shape0atrain/gradients/pi/Normal/prob/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
á
,atrain/gradients/pi/Normal/prob/sub_grad/SumSum,atrain/gradients/pi/Normal/prob/Exp_grad/mul>atrain/gradients/pi/Normal/prob/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ů
0atrain/gradients/pi/Normal/prob/sub_grad/ReshapeReshape,atrain/gradients/pi/Normal/prob/sub_grad/Sum.atrain/gradients/pi/Normal/prob/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
.atrain/gradients/pi/Normal/prob/sub_grad/Sum_1Sum,atrain/gradients/pi/Normal/prob/Exp_grad/mul@atrain/gradients/pi/Normal/prob/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

,atrain/gradients/pi/Normal/prob/sub_grad/NegNeg.atrain/gradients/pi/Normal/prob/sub_grad/Sum_1*
T0*
_output_shapes
:
Ý
2atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1Reshape,atrain/gradients/pi/Normal/prob/sub_grad/Neg0atrain/gradients/pi/Normal/prob/sub_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Š
9atrain/gradients/pi/Normal/prob/sub_grad/tuple/group_depsNoOp1^atrain/gradients/pi/Normal/prob/sub_grad/Reshape3^atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1
˛
Aatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependencyIdentity0atrain/gradients/pi/Normal/prob/sub_grad/Reshape:^atrain/gradients/pi/Normal/prob/sub_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*C
_class9
75loc:@atrain/gradients/pi/Normal/prob/sub_grad/Reshape
¸
Catrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1:^atrain/gradients/pi/Normal/prob/sub_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
.atrain/gradients/pi/Normal/prob/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

0atrain/gradients/pi/Normal/prob/mul_grad/Shape_1Shapepi/Normal/prob/Square*
_output_shapes
:*
T0*
out_type0
ö
>atrain/gradients/pi/Normal/prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/pi/Normal/prob/mul_grad/Shape0atrain/gradients/pi/Normal/prob/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ż
,atrain/gradients/pi/Normal/prob/mul_grad/MulMulAatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependencypi/Normal/prob/Square*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
,atrain/gradients/pi/Normal/prob/mul_grad/SumSum,atrain/gradients/pi/Normal/prob/mul_grad/Mul>atrain/gradients/pi/Normal/prob/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Č
0atrain/gradients/pi/Normal/prob/mul_grad/ReshapeReshape,atrain/gradients/pi/Normal/prob/mul_grad/Sum.atrain/gradients/pi/Normal/prob/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ŕ
.atrain/gradients/pi/Normal/prob/mul_grad/Mul_1Mulpi/Normal/prob/mul/xAatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ç
.atrain/gradients/pi/Normal/prob/mul_grad/Sum_1Sum.atrain/gradients/pi/Normal/prob/mul_grad/Mul_1@atrain/gradients/pi/Normal/prob/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ß
2atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1Reshape.atrain/gradients/pi/Normal/prob/mul_grad/Sum_10atrain/gradients/pi/Normal/prob/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
9atrain/gradients/pi/Normal/prob/mul_grad/tuple/group_depsNoOp1^atrain/gradients/pi/Normal/prob/mul_grad/Reshape3^atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1
Ą
Aatrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependencyIdentity0atrain/gradients/pi/Normal/prob/mul_grad/Reshape:^atrain/gradients/pi/Normal/prob/mul_grad/tuple/group_deps*
T0*C
_class9
75loc:@atrain/gradients/pi/Normal/prob/mul_grad/Reshape*
_output_shapes
: 
¸
Catrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1:^atrain/gradients/pi/Normal/prob/mul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*E
_class;
97loc:@atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1
q
.atrain/gradients/pi/Normal/prob/add_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 

0atrain/gradients/pi/Normal/prob/add_grad/Shape_1Shapepi/Normal/prob/Log*
T0*
out_type0*
_output_shapes
:
ö
>atrain/gradients/pi/Normal/prob/add_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/pi/Normal/prob/add_grad/Shape0atrain/gradients/pi/Normal/prob/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ř
,atrain/gradients/pi/Normal/prob/add_grad/SumSumCatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency_1>atrain/gradients/pi/Normal/prob/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Č
0atrain/gradients/pi/Normal/prob/add_grad/ReshapeReshape,atrain/gradients/pi/Normal/prob/add_grad/Sum.atrain/gradients/pi/Normal/prob/add_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ü
.atrain/gradients/pi/Normal/prob/add_grad/Sum_1SumCatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency_1@atrain/gradients/pi/Normal/prob/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ß
2atrain/gradients/pi/Normal/prob/add_grad/Reshape_1Reshape.atrain/gradients/pi/Normal/prob/add_grad/Sum_10atrain/gradients/pi/Normal/prob/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
9atrain/gradients/pi/Normal/prob/add_grad/tuple/group_depsNoOp1^atrain/gradients/pi/Normal/prob/add_grad/Reshape3^atrain/gradients/pi/Normal/prob/add_grad/Reshape_1
Ą
Aatrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependencyIdentity0atrain/gradients/pi/Normal/prob/add_grad/Reshape:^atrain/gradients/pi/Normal/prob/add_grad/tuple/group_deps*
_output_shapes
: *
T0*C
_class9
75loc:@atrain/gradients/pi/Normal/prob/add_grad/Reshape
¸
Catrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/Normal/prob/add_grad/Reshape_1:^atrain/gradients/pi/Normal/prob/add_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/pi/Normal/prob/add_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
1atrain/gradients/pi/Normal/prob/Square_grad/ConstConstD^atrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
ż
/atrain/gradients/pi/Normal/prob/Square_grad/MulMul"pi/Normal/prob/standardize/truediv1atrain/gradients/pi/Normal/prob/Square_grad/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ŕ
1atrain/gradients/pi/Normal/prob/Square_grad/Mul_1MulCatrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependency_1/atrain/gradients/pi/Normal/prob/Square_grad/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
3atrain/gradients/pi/Normal/prob/Log_grad/Reciprocal
Reciprocalpi/Normal/scaleD^atrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
,atrain/gradients/pi/Normal/prob/Log_grad/mulMulCatrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependency_13atrain/gradients/pi/Normal/prob/Log_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ShapeShapepi/Normal/prob/standardize/sub*
T0*
out_type0*
_output_shapes
:

@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape_1Shapepi/Normal/scale*
T0*
out_type0*
_output_shapes
:
Ś
Natrain/gradients/pi/Normal/prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Á
@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDivRealDiv1atrain/gradients/pi/Normal/prob/Square_grad/Mul_1pi/Normal/scale*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/SumSum@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDivNatrain/gradients/pi/Normal/prob/standardize/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ReshapeReshape<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Sum>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/NegNegpi/Normal/prob/standardize/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Î
Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_1RealDiv<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Negpi/Normal/scale*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ô
Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_2RealDivBatrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_1pi/Normal/scale*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ě
<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/mulMul1atrain/gradients/pi/Normal/prob/Square_grad/Mul_1Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Sum_1Sum<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/mulPatrain/gradients/pi/Normal/prob/standardize/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1Reshape>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Sum_1@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ů
Iatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/group_depsNoOpA^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ReshapeC^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1
ň
Qatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependencyIdentity@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ReshapeJ^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/group_deps*
T0*S
_classI
GEloc:@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ř
Satrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependency_1IdentityBatrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1J^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*U
_classK
IGloc:@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1

:atrain/gradients/pi/Normal/prob/standardize/sub_grad/ShapeShapeaction*
_output_shapes
:*
T0*
out_type0

<atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape_1Shapepi/Normal/loc*
out_type0*
_output_shapes
:*
T0

Jatrain/gradients/pi/Normal/prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape<atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

8atrain/gradients/pi/Normal/prob/standardize/sub_grad/SumSumQatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependencyJatrain/gradients/pi/Normal/prob/standardize/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ý
<atrain/gradients/pi/Normal/prob/standardize/sub_grad/ReshapeReshape8atrain/gradients/pi/Normal/prob/standardize/sub_grad/Sum:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Sum_1SumQatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependencyLatrain/gradients/pi/Normal/prob/standardize/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

8atrain/gradients/pi/Normal/prob/standardize/sub_grad/NegNeg:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:

>atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1Reshape8atrain/gradients/pi/Normal/prob/standardize/sub_grad/Neg<atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
Eatrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/group_depsNoOp=^atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape?^atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1
â
Matrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/control_dependencyIdentity<atrain/gradients/pi/Normal/prob/standardize/sub_grad/ReshapeF^atrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/group_deps*
T0*O
_classE
CAloc:@atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
Oatrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/control_dependency_1Identity>atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1F^atrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

atrain/gradients/AddN_1AddN,atrain/gradients/pi/Normal/prob/Log_grad/mulSatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependency_1*
T0*?
_class5
31loc:@atrain/gradients/pi/Normal/prob/Log_grad/mul*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
+atrain/gradients/pi/scaled_sigma_grad/ShapeShapepi/dense/Softplus*
T0*
out_type0*
_output_shapes
:
w
-atrain/gradients/pi/scaled_sigma_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
í
;atrain/gradients/pi/scaled_sigma_grad/BroadcastGradientArgsBroadcastGradientArgs+atrain/gradients/pi/scaled_sigma_grad/Shape-atrain/gradients/pi/scaled_sigma_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

)atrain/gradients/pi/scaled_sigma_grad/MulMulatrain/gradients/AddN_1pi/scaled_sigma/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ř
)atrain/gradients/pi/scaled_sigma_grad/SumSum)atrain/gradients/pi/scaled_sigma_grad/Mul;atrain/gradients/pi/scaled_sigma_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Đ
-atrain/gradients/pi/scaled_sigma_grad/ReshapeReshape)atrain/gradients/pi/scaled_sigma_grad/Sum+atrain/gradients/pi/scaled_sigma_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

+atrain/gradients/pi/scaled_sigma_grad/Mul_1Mulpi/dense/Softplusatrain/gradients/AddN_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ţ
+atrain/gradients/pi/scaled_sigma_grad/Sum_1Sum+atrain/gradients/pi/scaled_sigma_grad/Mul_1=atrain/gradients/pi/scaled_sigma_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
É
/atrain/gradients/pi/scaled_sigma_grad/Reshape_1Reshape+atrain/gradients/pi/scaled_sigma_grad/Sum_1-atrain/gradients/pi/scaled_sigma_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
 
6atrain/gradients/pi/scaled_sigma_grad/tuple/group_depsNoOp.^atrain/gradients/pi/scaled_sigma_grad/Reshape0^atrain/gradients/pi/scaled_sigma_grad/Reshape_1
Ś
>atrain/gradients/pi/scaled_sigma_grad/tuple/control_dependencyIdentity-atrain/gradients/pi/scaled_sigma_grad/Reshape7^atrain/gradients/pi/scaled_sigma_grad/tuple/group_deps*
T0*@
_class6
42loc:@atrain/gradients/pi/scaled_sigma_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

@atrain/gradients/pi/scaled_sigma_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/scaled_sigma_grad/Reshape_17^atrain/gradients/pi/scaled_sigma_grad/tuple/group_deps*B
_class8
64loc:@atrain/gradients/pi/scaled_sigma_grad/Reshape_1*
_output_shapes
:*
T0
q
(atrain/gradients/pi/scaled_mu_grad/ShapeShape	pi/a/Tanh*
T0*
out_type0*
_output_shapes
:
t
*atrain/gradients/pi/scaled_mu_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ä
8atrain/gradients/pi/scaled_mu_grad/BroadcastGradientArgsBroadcastGradientArgs(atrain/gradients/pi/scaled_mu_grad/Shape*atrain/gradients/pi/scaled_mu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ŕ
&atrain/gradients/pi/scaled_mu_grad/MulMulOatrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/control_dependency_1pi/scaled_mu/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ď
&atrain/gradients/pi/scaled_mu_grad/SumSum&atrain/gradients/pi/scaled_mu_grad/Mul8atrain/gradients/pi/scaled_mu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ç
*atrain/gradients/pi/scaled_mu_grad/ReshapeReshape&atrain/gradients/pi/scaled_mu_grad/Sum(atrain/gradients/pi/scaled_mu_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˝
(atrain/gradients/pi/scaled_mu_grad/Mul_1Mul	pi/a/TanhOatrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/control_dependency_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ő
(atrain/gradients/pi/scaled_mu_grad/Sum_1Sum(atrain/gradients/pi/scaled_mu_grad/Mul_1:atrain/gradients/pi/scaled_mu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ŕ
,atrain/gradients/pi/scaled_mu_grad/Reshape_1Reshape(atrain/gradients/pi/scaled_mu_grad/Sum_1*atrain/gradients/pi/scaled_mu_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0

3atrain/gradients/pi/scaled_mu_grad/tuple/group_depsNoOp+^atrain/gradients/pi/scaled_mu_grad/Reshape-^atrain/gradients/pi/scaled_mu_grad/Reshape_1

;atrain/gradients/pi/scaled_mu_grad/tuple/control_dependencyIdentity*atrain/gradients/pi/scaled_mu_grad/Reshape4^atrain/gradients/pi/scaled_mu_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*=
_class3
1/loc:@atrain/gradients/pi/scaled_mu_grad/Reshape

=atrain/gradients/pi/scaled_mu_grad/tuple/control_dependency_1Identity,atrain/gradients/pi/scaled_mu_grad/Reshape_14^atrain/gradients/pi/scaled_mu_grad/tuple/group_deps*
T0*?
_class5
31loc:@atrain/gradients/pi/scaled_mu_grad/Reshape_1*
_output_shapes
:
Č
4atrain/gradients/pi/dense/Softplus_grad/SoftplusGradSoftplusGrad>atrain/gradients/pi/scaled_sigma_grad/tuple/control_dependencypi/dense/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
(atrain/gradients/pi/a/Tanh_grad/TanhGradTanhGrad	pi/a/Tanh;atrain/gradients/pi/scaled_mu_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
2atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad4atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad*
T0*
data_formatNHWC*
_output_shapes
:
Ť
7atrain/gradients/pi/dense/BiasAdd_grad/tuple/group_depsNoOp3^atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad
ś
?atrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity4atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad8^atrain/gradients/pi/dense/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*G
_class=
;9loc:@atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad
§
Aatrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGrad8^atrain/gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*E
_class;
97loc:@atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGrad
Ł
.atrain/gradients/pi/a/BiasAdd_grad/BiasAddGradBiasAddGrad(atrain/gradients/pi/a/Tanh_grad/TanhGrad*
_output_shapes
:*
T0*
data_formatNHWC

3atrain/gradients/pi/a/BiasAdd_grad/tuple/group_depsNoOp/^atrain/gradients/pi/a/BiasAdd_grad/BiasAddGrad)^atrain/gradients/pi/a/Tanh_grad/TanhGrad

;atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependencyIdentity(atrain/gradients/pi/a/Tanh_grad/TanhGrad4^atrain/gradients/pi/a/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@atrain/gradients/pi/a/Tanh_grad/TanhGrad*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

=atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependency_1Identity.atrain/gradients/pi/a/BiasAdd_grad/BiasAddGrad4^atrain/gradients/pi/a/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@atrain/gradients/pi/a/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ć
,atrain/gradients/pi/dense/MatMul_grad/MatMulMatMul?atrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependencypi/dense/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ő
.atrain/gradients/pi/dense/MatMul_grad/MatMul_1MatMul
pi/l4/Relu?atrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0

6atrain/gradients/pi/dense/MatMul_grad/tuple/group_depsNoOp-^atrain/gradients/pi/dense/MatMul_grad/MatMul/^atrain/gradients/pi/dense/MatMul_grad/MatMul_1
Ľ
>atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependencyIdentity,atrain/gradients/pi/dense/MatMul_grad/MatMul7^atrain/gradients/pi/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@atrain/gradients/pi/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
@atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Identity.atrain/gradients/pi/dense/MatMul_grad/MatMul_17^atrain/gradients/pi/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@atrain/gradients/pi/dense/MatMul_grad/MatMul_1*
_output_shapes
:	
Ú
(atrain/gradients/pi/a/MatMul_grad/MatMulMatMul;atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependencypi/a/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Í
*atrain/gradients/pi/a/MatMul_grad/MatMul_1MatMul
pi/l4/Relu;atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 

2atrain/gradients/pi/a/MatMul_grad/tuple/group_depsNoOp)^atrain/gradients/pi/a/MatMul_grad/MatMul+^atrain/gradients/pi/a/MatMul_grad/MatMul_1

:atrain/gradients/pi/a/MatMul_grad/tuple/control_dependencyIdentity(atrain/gradients/pi/a/MatMul_grad/MatMul3^atrain/gradients/pi/a/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@atrain/gradients/pi/a/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<atrain/gradients/pi/a/MatMul_grad/tuple/control_dependency_1Identity*atrain/gradients/pi/a/MatMul_grad/MatMul_13^atrain/gradients/pi/a/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*=
_class3
1/loc:@atrain/gradients/pi/a/MatMul_grad/MatMul_1

atrain/gradients/AddN_2AddN>atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependency:atrain/gradients/pi/a/MatMul_grad/tuple/control_dependency*
T0*?
_class5
31loc:@atrain/gradients/pi/dense/MatMul_grad/MatMul*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

)atrain/gradients/pi/l4/Relu_grad/ReluGradReluGradatrain/gradients/AddN_2
pi/l4/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
/atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

4atrain/gradients/pi/l4/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l4/Relu_grad/ReluGrad

<atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l4/Relu_grad/ReluGrad5^atrain/gradients/pi/l4/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l4/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l4/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ý
)atrain/gradients/pi/l4/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependencypi/l4/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Đ
+atrain/gradients/pi/l4/MatMul_grad/MatMul_1MatMul
pi/l3/Relu<atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

3atrain/gradients/pi/l4/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l4/MatMul_grad/MatMul,^atrain/gradients/pi/l4/MatMul_grad/MatMul_1

;atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l4/MatMul_grad/MatMul4^atrain/gradients/pi/l4/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@atrain/gradients/pi/l4/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

=atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l4/MatMul_grad/MatMul_14^atrain/gradients/pi/l4/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@atrain/gradients/pi/l4/MatMul_grad/MatMul_1* 
_output_shapes
:

ą
)atrain/gradients/pi/l3/Relu_grad/ReluGradReluGrad;atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependency
pi/l3/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
/atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l3/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*
data_formatNHWC

4atrain/gradients/pi/l3/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l3/Relu_grad/ReluGrad

<atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l3/Relu_grad/ReluGrad5^atrain/gradients/pi/l3/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l3/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l3/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*B
_class8
64loc:@atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGrad
Ý
)atrain/gradients/pi/l3/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependencypi/l3/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Đ
+atrain/gradients/pi/l3/MatMul_grad/MatMul_1MatMul
pi/l2/Relu<atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

3atrain/gradients/pi/l3/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l3/MatMul_grad/MatMul,^atrain/gradients/pi/l3/MatMul_grad/MatMul_1

;atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l3/MatMul_grad/MatMul4^atrain/gradients/pi/l3/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l3/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l3/MatMul_grad/MatMul_14^atrain/gradients/pi/l3/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@atrain/gradients/pi/l3/MatMul_grad/MatMul_1* 
_output_shapes
:

ą
)atrain/gradients/pi/l2/Relu_grad/ReluGradReluGrad;atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependency
pi/l2/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
/atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

4atrain/gradients/pi/l2/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l2/Relu_grad/ReluGrad

<atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l2/Relu_grad/ReluGrad5^atrain/gradients/pi/l2/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l2/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l2/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*B
_class8
64loc:@atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGrad
Ý
)atrain/gradients/pi/l2/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependencypi/l2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Đ
+atrain/gradients/pi/l2/MatMul_grad/MatMul_1MatMul
pi/l1/Relu<atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(

3atrain/gradients/pi/l2/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l2/MatMul_grad/MatMul,^atrain/gradients/pi/l2/MatMul_grad/MatMul_1

;atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l2/MatMul_grad/MatMul4^atrain/gradients/pi/l2/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l2/MatMul_grad/MatMul_14^atrain/gradients/pi/l2/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@atrain/gradients/pi/l2/MatMul_grad/MatMul_1* 
_output_shapes
:

ą
)atrain/gradients/pi/l1/Relu_grad/ReluGradReluGrad;atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependency
pi/l1/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
/atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

4atrain/gradients/pi/l1/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l1/Relu_grad/ReluGrad

<atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l1/Relu_grad/ReluGrad5^atrain/gradients/pi/l1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l1/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l1/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ü
)atrain/gradients/pi/l1/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependencypi/l1/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Ę
+atrain/gradients/pi/l1/MatMul_grad/MatMul_1MatMulstate<atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 

3atrain/gradients/pi/l1/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l1/MatMul_grad/MatMul,^atrain/gradients/pi/l1/MatMul_grad/MatMul_1

;atrain/gradients/pi/l1/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l1/MatMul_grad/MatMul4^atrain/gradients/pi/l1/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@atrain/gradients/pi/l1/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

=atrain/gradients/pi/l1/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l1/MatMul_grad/MatMul_14^atrain/gradients/pi/l1/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@atrain/gradients/pi/l1/MatMul_grad/MatMul_1*
_output_shapes
:	

 atrain/beta1_power/initial_valueConst*
_class
loc:@pi/a/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 

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
Á
atrain/beta1_power/AssignAssignatrain/beta1_power atrain/beta1_power/initial_value*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
v
atrain/beta1_power/readIdentityatrain/beta1_power*
_output_shapes
: *
T0*
_class
loc:@pi/a/bias

 atrain/beta2_power/initial_valueConst*
_class
loc:@pi/a/bias*
valueB
 *wž?*
dtype0*
_output_shapes
: 

atrain/beta2_power
VariableV2*
_class
loc:@pi/a/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Á
atrain/beta2_power/AssignAssignatrain/beta2_power atrain/beta2_power/initial_value*
use_locking(*
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
: 
v
atrain/beta2_power/readIdentityatrain/beta2_power*
T0*
_class
loc:@pi/a/bias*
_output_shapes
: 
Ź
:atrain/pi/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l1/kernel*
valueB"      *
dtype0*
_output_shapes
:

0atrain/pi/l1/kernel/Adam/Initializer/zeros/ConstConst*
_class
loc:@pi/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ý
*atrain/pi/l1/kernel/Adam/Initializer/zerosFill:atrain/pi/l1/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l1/kernel/Adam/Initializer/zeros/Const*
T0*
_class
loc:@pi/l1/kernel*

index_type0*
_output_shapes
:	
Ż
atrain/pi/l1/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *
_class
loc:@pi/l1/kernel*
	container *
shape:	
ă
atrain/pi/l1/kernel/Adam/AssignAssignatrain/pi/l1/kernel/Adam*atrain/pi/l1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l1/kernel*
validate_shape(*
_output_shapes
:	

atrain/pi/l1/kernel/Adam/readIdentityatrain/pi/l1/kernel/Adam*
T0*
_class
loc:@pi/l1/kernel*
_output_shapes
:	
Ž
<atrain/pi/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@pi/l1/kernel*
valueB"      

2atrain/pi/l1/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@pi/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,atrain/pi/l1/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l1/kernel/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@pi/l1/kernel*

index_type0*
_output_shapes
:	
ą
atrain/pi/l1/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *
_class
loc:@pi/l1/kernel*
	container *
shape:	
é
!atrain/pi/l1/kernel/Adam_1/AssignAssignatrain/pi/l1/kernel/Adam_1,atrain/pi/l1/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@pi/l1/kernel*
validate_shape(

atrain/pi/l1/kernel/Adam_1/readIdentityatrain/pi/l1/kernel/Adam_1*
T0*
_class
loc:@pi/l1/kernel*
_output_shapes
:	

(atrain/pi/l1/bias/Adam/Initializer/zerosConst*
_class
loc:@pi/l1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ł
atrain/pi/l1/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l1/bias*
	container 
×
atrain/pi/l1/bias/Adam/AssignAssignatrain/pi/l1/bias/Adam(atrain/pi/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l1/bias*
validate_shape(*
_output_shapes	
:

atrain/pi/l1/bias/Adam/readIdentityatrain/pi/l1/bias/Adam*
T0*
_class
loc:@pi/l1/bias*
_output_shapes	
:

*atrain/pi/l1/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ľ
atrain/pi/l1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l1/bias*
	container *
shape:
Ý
atrain/pi/l1/bias/Adam_1/AssignAssignatrain/pi/l1/bias/Adam_1*atrain/pi/l1/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@pi/l1/bias*
validate_shape(

atrain/pi/l1/bias/Adam_1/readIdentityatrain/pi/l1/bias/Adam_1*
_class
loc:@pi/l1/bias*
_output_shapes	
:*
T0
Ź
:atrain/pi/l2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:

0atrain/pi/l2/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@pi/l2/kernel*
valueB
 *    
ţ
*atrain/pi/l2/kernel/Adam/Initializer/zerosFill:atrain/pi/l2/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l2/kernel/Adam/Initializer/zeros/Const*
T0*
_class
loc:@pi/l2/kernel*

index_type0* 
_output_shapes
:

ą
atrain/pi/l2/kernel/Adam
VariableV2* 
_output_shapes
:
*
shared_name *
_class
loc:@pi/l2/kernel*
	container *
shape:
*
dtype0
ä
atrain/pi/l2/kernel/Adam/AssignAssignatrain/pi/l2/kernel/Adam*atrain/pi/l2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l2/kernel*
validate_shape(* 
_output_shapes
:


atrain/pi/l2/kernel/Adam/readIdentityatrain/pi/l2/kernel/Adam*
T0*
_class
loc:@pi/l2/kernel* 
_output_shapes
:

Ž
<atrain/pi/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:

2atrain/pi/l2/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@pi/l2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,atrain/pi/l2/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l2/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*
_class
loc:@pi/l2/kernel*

index_type0
ł
atrain/pi/l2/kernel/Adam_1
VariableV2*
_class
loc:@pi/l2/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
ę
!atrain/pi/l2/kernel/Adam_1/AssignAssignatrain/pi/l2/kernel/Adam_1,atrain/pi/l2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l2/kernel*
validate_shape(* 
_output_shapes
:


atrain/pi/l2/kernel/Adam_1/readIdentityatrain/pi/l2/kernel/Adam_1*
T0*
_class
loc:@pi/l2/kernel* 
_output_shapes
:


(atrain/pi/l2/bias/Adam/Initializer/zerosConst*
_class
loc:@pi/l2/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ł
atrain/pi/l2/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l2/bias*
	container 
×
atrain/pi/l2/bias/Adam/AssignAssignatrain/pi/l2/bias/Adam(atrain/pi/l2/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l2/bias*
validate_shape(*
_output_shapes	
:

atrain/pi/l2/bias/Adam/readIdentityatrain/pi/l2/bias/Adam*
_output_shapes	
:*
T0*
_class
loc:@pi/l2/bias

*atrain/pi/l2/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l2/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ľ
atrain/pi/l2/bias/Adam_1
VariableV2*
_class
loc:@pi/l2/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ý
atrain/pi/l2/bias/Adam_1/AssignAssignatrain/pi/l2/bias/Adam_1*atrain/pi/l2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l2/bias*
validate_shape(*
_output_shapes	
:

atrain/pi/l2/bias/Adam_1/readIdentityatrain/pi/l2/bias/Adam_1*
T0*
_class
loc:@pi/l2/bias*
_output_shapes	
:
Ź
:atrain/pi/l3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:

0atrain/pi/l3/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
_class
loc:@pi/l3/kernel*
valueB
 *    *
dtype0
ţ
*atrain/pi/l3/kernel/Adam/Initializer/zerosFill:atrain/pi/l3/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l3/kernel/Adam/Initializer/zeros/Const*
T0*
_class
loc:@pi/l3/kernel*

index_type0* 
_output_shapes
:

ą
atrain/pi/l3/kernel/Adam
VariableV2*
_class
loc:@pi/l3/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
ä
atrain/pi/l3/kernel/Adam/AssignAssignatrain/pi/l3/kernel/Adam*atrain/pi/l3/kernel/Adam/Initializer/zeros*
_class
loc:@pi/l3/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

atrain/pi/l3/kernel/Adam/readIdentityatrain/pi/l3/kernel/Adam*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:

Ž
<atrain/pi/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
_class
loc:@pi/l3/kernel*
valueB"      *
dtype0

2atrain/pi/l3/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
_class
loc:@pi/l3/kernel*
valueB
 *    *
dtype0

,atrain/pi/l3/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l3/kernel/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@pi/l3/kernel*

index_type0* 
_output_shapes
:

ł
atrain/pi/l3/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *
_class
loc:@pi/l3/kernel*
	container *
shape:

ę
!atrain/pi/l3/kernel/Adam_1/AssignAssignatrain/pi/l3/kernel/Adam_1,atrain/pi/l3/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l3/kernel*
validate_shape(* 
_output_shapes
:


atrain/pi/l3/kernel/Adam_1/readIdentityatrain/pi/l3/kernel/Adam_1*
_class
loc:@pi/l3/kernel* 
_output_shapes
:
*
T0

(atrain/pi/l3/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*
_class
loc:@pi/l3/bias*
valueB*    *
dtype0
Ł
atrain/pi/l3/bias/Adam
VariableV2*
_class
loc:@pi/l3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
×
atrain/pi/l3/bias/Adam/AssignAssignatrain/pi/l3/bias/Adam(atrain/pi/l3/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l3/bias*
validate_shape(*
_output_shapes	
:

atrain/pi/l3/bias/Adam/readIdentityatrain/pi/l3/bias/Adam*
T0*
_class
loc:@pi/l3/bias*
_output_shapes	
:

*atrain/pi/l3/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l3/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ľ
atrain/pi/l3/bias/Adam_1
VariableV2*
_class
loc:@pi/l3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ý
atrain/pi/l3/bias/Adam_1/AssignAssignatrain/pi/l3/bias/Adam_1*atrain/pi/l3/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l3/bias*
validate_shape(*
_output_shapes	
:

atrain/pi/l3/bias/Adam_1/readIdentityatrain/pi/l3/bias/Adam_1*
T0*
_class
loc:@pi/l3/bias*
_output_shapes	
:
Ź
:atrain/pi/l4/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l4/kernel*
valueB"      *
dtype0*
_output_shapes
:

0atrain/pi/l4/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@pi/l4/kernel*
valueB
 *    
ţ
*atrain/pi/l4/kernel/Adam/Initializer/zerosFill:atrain/pi/l4/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l4/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*
T0*
_class
loc:@pi/l4/kernel*

index_type0
ą
atrain/pi/l4/kernel/Adam
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *
_class
loc:@pi/l4/kernel
ä
atrain/pi/l4/kernel/Adam/AssignAssignatrain/pi/l4/kernel/Adam*atrain/pi/l4/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l4/kernel*
validate_shape(* 
_output_shapes
:


atrain/pi/l4/kernel/Adam/readIdentityatrain/pi/l4/kernel/Adam*
T0*
_class
loc:@pi/l4/kernel* 
_output_shapes
:

Ž
<atrain/pi/l4/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@pi/l4/kernel*
valueB"      

2atrain/pi/l4/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@pi/l4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,atrain/pi/l4/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l4/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l4/kernel/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@pi/l4/kernel*

index_type0* 
_output_shapes
:

ł
atrain/pi/l4/kernel/Adam_1
VariableV2*
_class
loc:@pi/l4/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
ę
!atrain/pi/l4/kernel/Adam_1/AssignAssignatrain/pi/l4/kernel/Adam_1,atrain/pi/l4/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@pi/l4/kernel

atrain/pi/l4/kernel/Adam_1/readIdentityatrain/pi/l4/kernel/Adam_1*
T0*
_class
loc:@pi/l4/kernel* 
_output_shapes
:


(atrain/pi/l4/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
_class
loc:@pi/l4/bias*
valueB*    
Ł
atrain/pi/l4/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l4/bias*
	container *
shape:
×
atrain/pi/l4/bias/Adam/AssignAssignatrain/pi/l4/bias/Adam(atrain/pi/l4/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l4/bias*
validate_shape(*
_output_shapes	
:

atrain/pi/l4/bias/Adam/readIdentityatrain/pi/l4/bias/Adam*
T0*
_class
loc:@pi/l4/bias*
_output_shapes	
:

*atrain/pi/l4/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l4/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ľ
atrain/pi/l4/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@pi/l4/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ý
atrain/pi/l4/bias/Adam_1/AssignAssignatrain/pi/l4/bias/Adam_1*atrain/pi/l4/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l4/bias*
validate_shape(*
_output_shapes	
:

atrain/pi/l4/bias/Adam_1/readIdentityatrain/pi/l4/bias/Adam_1*
_output_shapes	
:*
T0*
_class
loc:@pi/l4/bias
 
)atrain/pi/a/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	*
_class
loc:@pi/a/kernel*
valueB	*    
­
atrain/pi/a/kernel/Adam
VariableV2*
shared_name *
_class
loc:@pi/a/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
ß
atrain/pi/a/kernel/Adam/AssignAssignatrain/pi/a/kernel/Adam)atrain/pi/a/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/a/kernel*
validate_shape(*
_output_shapes
:	

atrain/pi/a/kernel/Adam/readIdentityatrain/pi/a/kernel/Adam*
T0*
_class
loc:@pi/a/kernel*
_output_shapes
:	
˘
+atrain/pi/a/kernel/Adam_1/Initializer/zerosConst*
_class
loc:@pi/a/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Ż
atrain/pi/a/kernel/Adam_1
VariableV2*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name *
_class
loc:@pi/a/kernel
ĺ
 atrain/pi/a/kernel/Adam_1/AssignAssignatrain/pi/a/kernel/Adam_1+atrain/pi/a/kernel/Adam_1/Initializer/zeros*
T0*
_class
loc:@pi/a/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(

atrain/pi/a/kernel/Adam_1/readIdentityatrain/pi/a/kernel/Adam_1*
T0*
_class
loc:@pi/a/kernel*
_output_shapes
:	

'atrain/pi/a/bias/Adam/Initializer/zerosConst*
_class
loc:@pi/a/bias*
valueB*    *
dtype0*
_output_shapes
:

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
Ň
atrain/pi/a/bias/Adam/AssignAssignatrain/pi/a/bias/Adam'atrain/pi/a/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/a/bias

atrain/pi/a/bias/Adam/readIdentityatrain/pi/a/bias/Adam*
T0*
_class
loc:@pi/a/bias*
_output_shapes
:

)atrain/pi/a/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/a/bias*
valueB*    *
dtype0*
_output_shapes
:
Ą
atrain/pi/a/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@pi/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ř
atrain/pi/a/bias/Adam_1/AssignAssignatrain/pi/a/bias/Adam_1)atrain/pi/a/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
:

atrain/pi/a/bias/Adam_1/readIdentityatrain/pi/a/bias/Adam_1*
T0*
_class
loc:@pi/a/bias*
_output_shapes
:
¨
-atrain/pi/dense/kernel/Adam/Initializer/zerosConst*
_output_shapes
:	*"
_class
loc:@pi/dense/kernel*
valueB	*    *
dtype0
ľ
atrain/pi/dense/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	
ď
"atrain/pi/dense/kernel/Adam/AssignAssignatrain/pi/dense/kernel/Adam-atrain/pi/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	

 atrain/pi/dense/kernel/Adam/readIdentityatrain/pi/dense/kernel/Adam*
_output_shapes
:	*
T0*"
_class
loc:@pi/dense/kernel
Ş
/atrain/pi/dense/kernel/Adam_1/Initializer/zerosConst*"
_class
loc:@pi/dense/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
ˇ
atrain/pi/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	
ő
$atrain/pi/dense/kernel/Adam_1/AssignAssignatrain/pi/dense/kernel/Adam_1/atrain/pi/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel

"atrain/pi/dense/kernel/Adam_1/readIdentityatrain/pi/dense/kernel/Adam_1*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	

+atrain/pi/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
§
atrain/pi/dense/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@pi/dense/bias
â
 atrain/pi/dense/bias/Adam/AssignAssignatrain/pi/dense/bias/Adam+atrain/pi/dense/bias/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(

atrain/pi/dense/bias/Adam/readIdentityatrain/pi/dense/bias/Adam*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:

-atrain/pi/dense/bias/Adam_1/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Š
atrain/pi/dense/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@pi/dense/bias
č
"atrain/pi/dense/bias/Adam_1/AssignAssignatrain/pi/dense/bias/Adam_1-atrain/pi/dense/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(

 atrain/pi/dense/bias/Adam_1/readIdentityatrain/pi/dense/bias/Adam_1*
_output_shapes
:*
T0* 
_class
loc:@pi/dense/bias
^
atrain/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *ˇŃ8
V
atrain/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
V
atrain/Adam/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
X
atrain/Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
ł
)atrain/Adam/update_pi/l1/kernel/ApplyAdam	ApplyAdampi/l1/kernelatrain/pi/l1/kernel/Adamatrain/pi/l1/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l1/MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@pi/l1/kernel*
use_nesterov( *
_output_shapes
:	*
use_locking( 
Ś
'atrain/Adam/update_pi/l1/bias/ApplyAdam	ApplyAdam
pi/l1/biasatrain/pi/l1/bias/Adamatrain/pi/l1/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l1/bias*
use_nesterov( *
_output_shapes	
:
´
)atrain/Adam/update_pi/l2/kernel/ApplyAdam	ApplyAdampi/l2/kernelatrain/pi/l2/kernel/Adamatrain/pi/l2/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0*
_class
loc:@pi/l2/kernel
Ś
'atrain/Adam/update_pi/l2/bias/ApplyAdam	ApplyAdam
pi/l2/biasatrain/pi/l2/bias/Adamatrain/pi/l2/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l2/bias*
use_nesterov( *
_output_shapes	
:
´
)atrain/Adam/update_pi/l3/kernel/ApplyAdam	ApplyAdampi/l3/kernelatrain/pi/l3/kernel/Adamatrain/pi/l3/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependency_1*
_class
loc:@pi/l3/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
Ś
'atrain/Adam/update_pi/l3/bias/ApplyAdam	ApplyAdam
pi/l3/biasatrain/pi/l3/bias/Adamatrain/pi/l3/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l3/bias*
use_nesterov( *
_output_shapes	
:
´
)atrain/Adam/update_pi/l4/kernel/ApplyAdam	ApplyAdampi/l4/kernelatrain/pi/l4/kernel/Adamatrain/pi/l4/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l4/kernel*
use_nesterov( * 
_output_shapes
:

Ś
'atrain/Adam/update_pi/l4/bias/ApplyAdam	ApplyAdam
pi/l4/biasatrain/pi/l4/bias/Adamatrain/pi/l4/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*
_class
loc:@pi/l4/bias
­
(atrain/Adam/update_pi/a/kernel/ApplyAdam	ApplyAdampi/a/kernelatrain/pi/a/kernel/Adamatrain/pi/a/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon<atrain/gradients/pi/a/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/a/kernel*
use_nesterov( *
_output_shapes
:	

&atrain/Adam/update_pi/a/bias/ApplyAdam	ApplyAdam	pi/a/biasatrain/pi/a/bias/Adamatrain/pi/a/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/a/bias*
use_nesterov( *
_output_shapes
:
Ĺ
,atrain/Adam/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelatrain/pi/dense/kernel/Adamatrain/pi/dense/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon@atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	*
use_locking( *
T0*"
_class
loc:@pi/dense/kernel*
use_nesterov( 
ˇ
*atrain/Adam/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biasatrain/pi/dense/bias/Adamatrain/pi/dense/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilonAatrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0* 
_class
loc:@pi/dense/bias*
use_nesterov( 

atrain/Adam/mulMulatrain/beta1_power/readatrain/Adam/beta1'^atrain/Adam/update_pi/a/bias/ApplyAdam)^atrain/Adam/update_pi/a/kernel/ApplyAdam+^atrain/Adam/update_pi/dense/bias/ApplyAdam-^atrain/Adam/update_pi/dense/kernel/ApplyAdam(^atrain/Adam/update_pi/l1/bias/ApplyAdam*^atrain/Adam/update_pi/l1/kernel/ApplyAdam(^atrain/Adam/update_pi/l2/bias/ApplyAdam*^atrain/Adam/update_pi/l2/kernel/ApplyAdam(^atrain/Adam/update_pi/l3/bias/ApplyAdam*^atrain/Adam/update_pi/l3/kernel/ApplyAdam(^atrain/Adam/update_pi/l4/bias/ApplyAdam*^atrain/Adam/update_pi/l4/kernel/ApplyAdam*
T0*
_class
loc:@pi/a/bias*
_output_shapes
: 
Š
atrain/Adam/AssignAssignatrain/beta1_poweratrain/Adam/mul*
use_locking( *
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
: 

atrain/Adam/mul_1Mulatrain/beta2_power/readatrain/Adam/beta2'^atrain/Adam/update_pi/a/bias/ApplyAdam)^atrain/Adam/update_pi/a/kernel/ApplyAdam+^atrain/Adam/update_pi/dense/bias/ApplyAdam-^atrain/Adam/update_pi/dense/kernel/ApplyAdam(^atrain/Adam/update_pi/l1/bias/ApplyAdam*^atrain/Adam/update_pi/l1/kernel/ApplyAdam(^atrain/Adam/update_pi/l2/bias/ApplyAdam*^atrain/Adam/update_pi/l2/kernel/ApplyAdam(^atrain/Adam/update_pi/l3/bias/ApplyAdam*^atrain/Adam/update_pi/l3/kernel/ApplyAdam(^atrain/Adam/update_pi/l4/bias/ApplyAdam*^atrain/Adam/update_pi/l4/kernel/ApplyAdam*
T0*
_class
loc:@pi/a/bias*
_output_shapes
: 
­
atrain/Adam/Assign_1Assignatrain/beta2_poweratrain/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@pi/a/bias
Ç
atrain/AdamNoOp^atrain/Adam/Assign^atrain/Adam/Assign_1'^atrain/Adam/update_pi/a/bias/ApplyAdam)^atrain/Adam/update_pi/a/kernel/ApplyAdam+^atrain/Adam/update_pi/dense/bias/ApplyAdam-^atrain/Adam/update_pi/dense/kernel/ApplyAdam(^atrain/Adam/update_pi/l1/bias/ApplyAdam*^atrain/Adam/update_pi/l1/kernel/ApplyAdam(^atrain/Adam/update_pi/l2/bias/ApplyAdam*^atrain/Adam/update_pi/l2/kernel/ApplyAdam(^atrain/Adam/update_pi/l3/bias/ApplyAdam*^atrain/Adam/update_pi/l3/kernel/ApplyAdam(^atrain/Adam/update_pi/l4/bias/ApplyAdam*^atrain/Adam/update_pi/l4/kernel/ApplyAdam")žŮbý     nęiç	}tđ	×AJŐú
Î Ž 
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
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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

2	

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

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
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
ö
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

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
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *1.9.02v1.9.0-0-g25c197e023Îü

h
statePlaceholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙

,critic/w1_s/Initializer/random_uniform/shapeConst*
_class
loc:@critic/w1_s*
valueB"      *
dtype0*
_output_shapes
:

*critic/w1_s/Initializer/random_uniform/minConst*
_class
loc:@critic/w1_s*
valueB
 *°îž*
dtype0*
_output_shapes
: 

*critic/w1_s/Initializer/random_uniform/maxConst*
_class
loc:@critic/w1_s*
valueB
 *°î>*
dtype0*
_output_shapes
: 
ă
4critic/w1_s/Initializer/random_uniform/RandomUniformRandomUniform,critic/w1_s/Initializer/random_uniform/shape*
T0*
_class
loc:@critic/w1_s*
seed2 *
dtype0*
_output_shapes
:	*

seed 
Ę
*critic/w1_s/Initializer/random_uniform/subSub*critic/w1_s/Initializer/random_uniform/max*critic/w1_s/Initializer/random_uniform/min*
T0*
_class
loc:@critic/w1_s*
_output_shapes
: 
Ý
*critic/w1_s/Initializer/random_uniform/mulMul4critic/w1_s/Initializer/random_uniform/RandomUniform*critic/w1_s/Initializer/random_uniform/sub*
T0*
_class
loc:@critic/w1_s*
_output_shapes
:	
Ď
&critic/w1_s/Initializer/random_uniformAdd*critic/w1_s/Initializer/random_uniform/mul*critic/w1_s/Initializer/random_uniform/min*
T0*
_class
loc:@critic/w1_s*
_output_shapes
:	
Ą
critic/w1_s
VariableV2*
_output_shapes
:	*
shared_name *
_class
loc:@critic/w1_s*
	container *
shape:	*
dtype0
Ä
critic/w1_s/AssignAssigncritic/w1_s&critic/w1_s/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@critic/w1_s*
validate_shape(*
_output_shapes
:	
s
critic/w1_s/readIdentitycritic/w1_s*
T0*
_class
loc:@critic/w1_s*
_output_shapes
:	

*critic/b1/Initializer/random_uniform/shapeConst*
_class
loc:@critic/b1*
valueB"      *
dtype0*
_output_shapes
:

(critic/b1/Initializer/random_uniform/minConst*
_class
loc:@critic/b1*
valueB
 *Ivž*
dtype0*
_output_shapes
: 

(critic/b1/Initializer/random_uniform/maxConst*
_class
loc:@critic/b1*
valueB
 *Iv>*
dtype0*
_output_shapes
: 
Ý
2critic/b1/Initializer/random_uniform/RandomUniformRandomUniform*critic/b1/Initializer/random_uniform/shape*
T0*
_class
loc:@critic/b1*
seed2 *
dtype0*
_output_shapes
:	*

seed 
Â
(critic/b1/Initializer/random_uniform/subSub(critic/b1/Initializer/random_uniform/max(critic/b1/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@critic/b1
Ő
(critic/b1/Initializer/random_uniform/mulMul2critic/b1/Initializer/random_uniform/RandomUniform(critic/b1/Initializer/random_uniform/sub*
T0*
_class
loc:@critic/b1*
_output_shapes
:	
Ç
$critic/b1/Initializer/random_uniformAdd(critic/b1/Initializer/random_uniform/mul(critic/b1/Initializer/random_uniform/min*
T0*
_class
loc:@critic/b1*
_output_shapes
:	

	critic/b1
VariableV2*
shared_name *
_class
loc:@critic/b1*
	container *
shape:	*
dtype0*
_output_shapes
:	
ź
critic/b1/AssignAssign	critic/b1$critic/b1/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
:	
m
critic/b1/readIdentity	critic/b1*
T0*
_class
loc:@critic/b1*
_output_shapes
:	

critic/MatMulMatMulstatecritic/w1_s/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
c

critic/addAddcritic/MatMulcritic/b1/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
critic/ReluRelu
critic/add*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
§
1critic/l2/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@critic/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:

/critic/l2/kernel/Initializer/random_uniform/minConst*#
_class
loc:@critic/l2/kernel*
valueB
 *×łÝ˝*
dtype0*
_output_shapes
: 

/critic/l2/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@critic/l2/kernel*
valueB
 *×łÝ=*
dtype0*
_output_shapes
: 
ó
9critic/l2/kernel/Initializer/random_uniform/RandomUniformRandomUniform1critic/l2/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*#
_class
loc:@critic/l2/kernel*
seed2 
Ţ
/critic/l2/kernel/Initializer/random_uniform/subSub/critic/l2/kernel/Initializer/random_uniform/max/critic/l2/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@critic/l2/kernel*
_output_shapes
: 
ň
/critic/l2/kernel/Initializer/random_uniform/mulMul9critic/l2/kernel/Initializer/random_uniform/RandomUniform/critic/l2/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@critic/l2/kernel* 
_output_shapes
:

ä
+critic/l2/kernel/Initializer/random_uniformAdd/critic/l2/kernel/Initializer/random_uniform/mul/critic/l2/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@critic/l2/kernel* 
_output_shapes
:

­
critic/l2/kernel
VariableV2*
shared_name *#
_class
loc:@critic/l2/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ů
critic/l2/kernel/AssignAssigncritic/l2/kernel+critic/l2/kernel/Initializer/random_uniform*
T0*#
_class
loc:@critic/l2/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

critic/l2/kernel/readIdentitycritic/l2/kernel*
T0*#
_class
loc:@critic/l2/kernel* 
_output_shapes
:


 critic/l2/bias/Initializer/zerosConst*!
_class
loc:@critic/l2/bias*
valueB*    *
dtype0*
_output_shapes	
:

critic/l2/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *!
_class
loc:@critic/l2/bias
Ă
critic/l2/bias/AssignAssigncritic/l2/bias critic/l2/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@critic/l2/bias*
validate_shape(
x
critic/l2/bias/readIdentitycritic/l2/bias*!
_class
loc:@critic/l2/bias*
_output_shapes	
:*
T0

critic/l2/MatMulMatMulcritic/Relucritic/l2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

critic/l2/BiasAddBiasAddcritic/l2/MatMulcritic/l2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
critic/l2/ReluRelucritic/l2/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
1critic/l3/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@critic/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:

/critic/l3/kernel/Initializer/random_uniform/minConst*#
_class
loc:@critic/l3/kernel*
valueB
 *   ž*
dtype0*
_output_shapes
: 

/critic/l3/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@critic/l3/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
ó
9critic/l3/kernel/Initializer/random_uniform/RandomUniformRandomUniform1critic/l3/kernel/Initializer/random_uniform/shape*
T0*#
_class
loc:@critic/l3/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
Ţ
/critic/l3/kernel/Initializer/random_uniform/subSub/critic/l3/kernel/Initializer/random_uniform/max/critic/l3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@critic/l3/kernel
ň
/critic/l3/kernel/Initializer/random_uniform/mulMul9critic/l3/kernel/Initializer/random_uniform/RandomUniform/critic/l3/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*#
_class
loc:@critic/l3/kernel
ä
+critic/l3/kernel/Initializer/random_uniformAdd/critic/l3/kernel/Initializer/random_uniform/mul/critic/l3/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:

­
critic/l3/kernel
VariableV2*
shared_name *#
_class
loc:@critic/l3/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ů
critic/l3/kernel/AssignAssigncritic/l3/kernel+critic/l3/kernel/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@critic/l3/kernel*
validate_shape(* 
_output_shapes
:


critic/l3/kernel/readIdentitycritic/l3/kernel*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:


 critic/l3/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*!
_class
loc:@critic/l3/bias*
valueB*    

critic/l3/bias
VariableV2*
shared_name *!
_class
loc:@critic/l3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ă
critic/l3/bias/AssignAssigncritic/l3/bias critic/l3/bias/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@critic/l3/bias*
validate_shape(*
_output_shapes	
:
x
critic/l3/bias/readIdentitycritic/l3/bias*
_output_shapes	
:*
T0*!
_class
loc:@critic/l3/bias

critic/l3/MatMulMatMulcritic/l2/Relucritic/l3/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

critic/l3/BiasAddBiasAddcritic/l3/MatMulcritic/l3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
critic/l3/ReluRelucritic/l3/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
­
4critic/dense/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@critic/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:

2critic/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *&
_class
loc:@critic/dense/kernel*
valueB
 *n×\ž

2critic/dense/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@critic/dense/kernel*
valueB
 *n×\>*
dtype0*
_output_shapes
: 
ű
<critic/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform4critic/dense/kernel/Initializer/random_uniform/shape*&
_class
loc:@critic/dense/kernel*
seed2 *
dtype0*
_output_shapes
:	*

seed *
T0
ę
2critic/dense/kernel/Initializer/random_uniform/subSub2critic/dense/kernel/Initializer/random_uniform/max2critic/dense/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
: 
ý
2critic/dense/kernel/Initializer/random_uniform/mulMul<critic/dense/kernel/Initializer/random_uniform/RandomUniform2critic/dense/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	
ď
.critic/dense/kernel/Initializer/random_uniformAdd2critic/dense/kernel/Initializer/random_uniform/mul2critic/dense/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	
ą
critic/dense/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *&
_class
loc:@critic/dense/kernel*
	container *
shape:	
ä
critic/dense/kernel/AssignAssigncritic/dense/kernel.critic/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*&
_class
loc:@critic/dense/kernel*
validate_shape(*
_output_shapes
:	

critic/dense/kernel/readIdentitycritic/dense/kernel*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	

#critic/dense/bias/Initializer/zerosConst*$
_class
loc:@critic/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Ł
critic/dense/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@critic/dense/bias*
	container *
shape:
Î
critic/dense/bias/AssignAssigncritic/dense/bias#critic/dense/bias/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@critic/dense/bias*
validate_shape(*
_output_shapes
:

critic/dense/bias/readIdentitycritic/dense/bias*
_output_shapes
:*
T0*$
_class
loc:@critic/dense/bias

critic/dense/MatMulMatMulcritic/l3/Relucritic/dense/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

critic/dense/BiasAddBiasAddcritic/dense/MatMulcritic/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
critic/discounted_rPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙*
dtype0
n

critic/subSubcritic/discounted_rcritic/dense/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
critic/SquareSquare
critic/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
critic/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
n
critic/MeanMeancritic/Squarecritic/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Y
critic/gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
_
critic/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

critic/gradients/FillFillcritic/gradients/Shapecritic/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 

/critic/gradients/critic/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ł
)critic/gradients/critic/Mean_grad/ReshapeReshapecritic/gradients/Fill/critic/gradients/critic/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
t
'critic/gradients/critic/Mean_grad/ShapeShapecritic/Square*
_output_shapes
:*
T0*
out_type0
Ć
&critic/gradients/critic/Mean_grad/TileTile)critic/gradients/critic/Mean_grad/Reshape'critic/gradients/critic/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
)critic/gradients/critic/Mean_grad/Shape_1Shapecritic/Square*
T0*
out_type0*
_output_shapes
:
l
)critic/gradients/critic/Mean_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
q
'critic/gradients/critic/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ŕ
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
Ä
(critic/gradients/critic/Mean_grad/Prod_1Prod)critic/gradients/critic/Mean_grad/Shape_2)critic/gradients/critic/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
m
+critic/gradients/critic/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ź
)critic/gradients/critic/Mean_grad/MaximumMaximum(critic/gradients/critic/Mean_grad/Prod_1+critic/gradients/critic/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
Ş
*critic/gradients/critic/Mean_grad/floordivFloorDiv&critic/gradients/critic/Mean_grad/Prod)critic/gradients/critic/Mean_grad/Maximum*
T0*
_output_shapes
: 

&critic/gradients/critic/Mean_grad/CastCast*critic/gradients/critic/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
ś
)critic/gradients/critic/Mean_grad/truedivRealDiv&critic/gradients/critic/Mean_grad/Tile&critic/gradients/critic/Mean_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

)critic/gradients/critic/Square_grad/ConstConst*^critic/gradients/critic/Mean_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0

'critic/gradients/critic/Square_grad/MulMul
critic/sub)critic/gradients/critic/Square_grad/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ś
)critic/gradients/critic/Square_grad/Mul_1Mul)critic/gradients/critic/Mean_grad/truediv'critic/gradients/critic/Square_grad/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
&critic/gradients/critic/sub_grad/ShapeShapecritic/discounted_r*
_output_shapes
:*
T0*
out_type0
|
(critic/gradients/critic/sub_grad/Shape_1Shapecritic/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ţ
6critic/gradients/critic/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&critic/gradients/critic/sub_grad/Shape(critic/gradients/critic/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Î
$critic/gradients/critic/sub_grad/SumSum)critic/gradients/critic/Square_grad/Mul_16critic/gradients/critic/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Á
(critic/gradients/critic/sub_grad/ReshapeReshape$critic/gradients/critic/sub_grad/Sum&critic/gradients/critic/sub_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ň
&critic/gradients/critic/sub_grad/Sum_1Sum)critic/gradients/critic/Square_grad/Mul_18critic/gradients/critic/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
v
$critic/gradients/critic/sub_grad/NegNeg&critic/gradients/critic/sub_grad/Sum_1*
_output_shapes
:*
T0
Ĺ
*critic/gradients/critic/sub_grad/Reshape_1Reshape$critic/gradients/critic/sub_grad/Neg(critic/gradients/critic/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

1critic/gradients/critic/sub_grad/tuple/group_depsNoOp)^critic/gradients/critic/sub_grad/Reshape+^critic/gradients/critic/sub_grad/Reshape_1

9critic/gradients/critic/sub_grad/tuple/control_dependencyIdentity(critic/gradients/critic/sub_grad/Reshape2^critic/gradients/critic/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@critic/gradients/critic/sub_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

;critic/gradients/critic/sub_grad/tuple/control_dependency_1Identity*critic/gradients/critic/sub_grad/Reshape_12^critic/gradients/critic/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@critic/gradients/critic/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
6critic/gradients/critic/dense/BiasAdd_grad/BiasAddGradBiasAddGrad;critic/gradients/critic/sub_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0
ş
;critic/gradients/critic/dense/BiasAdd_grad/tuple/group_depsNoOp7^critic/gradients/critic/dense/BiasAdd_grad/BiasAddGrad<^critic/gradients/critic/sub_grad/tuple/control_dependency_1
ť
Ccritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependencyIdentity;critic/gradients/critic/sub_grad/tuple/control_dependency_1<^critic/gradients/critic/dense/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@critic/gradients/critic/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
Ecritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependency_1Identity6critic/gradients/critic/dense/BiasAdd_grad/BiasAddGrad<^critic/gradients/critic/dense/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@critic/gradients/critic/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ň
0critic/gradients/critic/dense/MatMul_grad/MatMulMatMulCcritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependencycritic/dense/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
á
2critic/gradients/critic/dense/MatMul_grad/MatMul_1MatMulcritic/l3/ReluCcritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
Ş
:critic/gradients/critic/dense/MatMul_grad/tuple/group_depsNoOp1^critic/gradients/critic/dense/MatMul_grad/MatMul3^critic/gradients/critic/dense/MatMul_grad/MatMul_1
ľ
Bcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependencyIdentity0critic/gradients/critic/dense/MatMul_grad/MatMul;^critic/gradients/critic/dense/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@critic/gradients/critic/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
Dcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependency_1Identity2critic/gradients/critic/dense/MatMul_grad/MatMul_1;^critic/gradients/critic/dense/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@critic/gradients/critic/dense/MatMul_grad/MatMul_1*
_output_shapes
:	
Ŕ
-critic/gradients/critic/l3/Relu_grad/ReluGradReluGradBcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependencycritic/l3/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
3critic/gradients/critic/l3/BiasAdd_grad/BiasAddGradBiasAddGrad-critic/gradients/critic/l3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ś
8critic/gradients/critic/l3/BiasAdd_grad/tuple/group_depsNoOp4^critic/gradients/critic/l3/BiasAdd_grad/BiasAddGrad.^critic/gradients/critic/l3/Relu_grad/ReluGrad
Ť
@critic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l3/Relu_grad/ReluGrad9^critic/gradients/critic/l3/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@critic/gradients/critic/l3/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
Bcritic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependency_1Identity3critic/gradients/critic/l3/BiasAdd_grad/BiasAddGrad9^critic/gradients/critic/l3/BiasAdd_grad/tuple/group_deps*F
_class<
:8loc:@critic/gradients/critic/l3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
é
-critic/gradients/critic/l3/MatMul_grad/MatMulMatMul@critic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependencycritic/l3/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ü
/critic/gradients/critic/l3/MatMul_grad/MatMul_1MatMulcritic/l2/Relu@critic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
Ą
7critic/gradients/critic/l3/MatMul_grad/tuple/group_depsNoOp.^critic/gradients/critic/l3/MatMul_grad/MatMul0^critic/gradients/critic/l3/MatMul_grad/MatMul_1
Š
?critic/gradients/critic/l3/MatMul_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l3/MatMul_grad/MatMul8^critic/gradients/critic/l3/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*@
_class6
42loc:@critic/gradients/critic/l3/MatMul_grad/MatMul
§
Acritic/gradients/critic/l3/MatMul_grad/tuple/control_dependency_1Identity/critic/gradients/critic/l3/MatMul_grad/MatMul_18^critic/gradients/critic/l3/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*B
_class8
64loc:@critic/gradients/critic/l3/MatMul_grad/MatMul_1
˝
-critic/gradients/critic/l2/Relu_grad/ReluGradReluGrad?critic/gradients/critic/l3/MatMul_grad/tuple/control_dependencycritic/l2/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
3critic/gradients/critic/l2/BiasAdd_grad/BiasAddGradBiasAddGrad-critic/gradients/critic/l2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
Ś
8critic/gradients/critic/l2/BiasAdd_grad/tuple/group_depsNoOp4^critic/gradients/critic/l2/BiasAdd_grad/BiasAddGrad.^critic/gradients/critic/l2/Relu_grad/ReluGrad
Ť
@critic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l2/Relu_grad/ReluGrad9^critic/gradients/critic/l2/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@critic/gradients/critic/l2/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
Bcritic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependency_1Identity3critic/gradients/critic/l2/BiasAdd_grad/BiasAddGrad9^critic/gradients/critic/l2/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*F
_class<
:8loc:@critic/gradients/critic/l2/BiasAdd_grad/BiasAddGrad
é
-critic/gradients/critic/l2/MatMul_grad/MatMulMatMul@critic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependencycritic/l2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Ů
/critic/gradients/critic/l2/MatMul_grad/MatMul_1MatMulcritic/Relu@critic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
Ą
7critic/gradients/critic/l2/MatMul_grad/tuple/group_depsNoOp.^critic/gradients/critic/l2/MatMul_grad/MatMul0^critic/gradients/critic/l2/MatMul_grad/MatMul_1
Š
?critic/gradients/critic/l2/MatMul_grad/tuple/control_dependencyIdentity-critic/gradients/critic/l2/MatMul_grad/MatMul8^critic/gradients/critic/l2/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*@
_class6
42loc:@critic/gradients/critic/l2/MatMul_grad/MatMul
§
Acritic/gradients/critic/l2/MatMul_grad/tuple/control_dependency_1Identity/critic/gradients/critic/l2/MatMul_grad/MatMul_18^critic/gradients/critic/l2/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*B
_class8
64loc:@critic/gradients/critic/l2/MatMul_grad/MatMul_1
ˇ
*critic/gradients/critic/Relu_grad/ReluGradReluGrad?critic/gradients/critic/l2/MatMul_grad/tuple/control_dependencycritic/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
&critic/gradients/critic/add_grad/ShapeShapecritic/MatMul*
T0*
out_type0*
_output_shapes
:
y
(critic/gradients/critic/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
Ţ
6critic/gradients/critic/add_grad/BroadcastGradientArgsBroadcastGradientArgs&critic/gradients/critic/add_grad/Shape(critic/gradients/critic/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ď
$critic/gradients/critic/add_grad/SumSum*critic/gradients/critic/Relu_grad/ReluGrad6critic/gradients/critic/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Â
(critic/gradients/critic/add_grad/ReshapeReshape$critic/gradients/critic/add_grad/Sum&critic/gradients/critic/add_grad/Shape*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ó
&critic/gradients/critic/add_grad/Sum_1Sum*critic/gradients/critic/Relu_grad/ReluGrad8critic/gradients/critic/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ż
*critic/gradients/critic/add_grad/Reshape_1Reshape&critic/gradients/critic/add_grad/Sum_1(critic/gradients/critic/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:	

1critic/gradients/critic/add_grad/tuple/group_depsNoOp)^critic/gradients/critic/add_grad/Reshape+^critic/gradients/critic/add_grad/Reshape_1

9critic/gradients/critic/add_grad/tuple/control_dependencyIdentity(critic/gradients/critic/add_grad/Reshape2^critic/gradients/critic/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@critic/gradients/critic/add_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

;critic/gradients/critic/add_grad/tuple/control_dependency_1Identity*critic/gradients/critic/add_grad/Reshape_12^critic/gradients/critic/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@critic/gradients/critic/add_grad/Reshape_1*
_output_shapes
:	
Ů
*critic/gradients/critic/MatMul_grad/MatMulMatMul9critic/gradients/critic/add_grad/tuple/control_dependencycritic/w1_s/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Č
,critic/gradients/critic/MatMul_grad/MatMul_1MatMulstate9critic/gradients/critic/add_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 

4critic/gradients/critic/MatMul_grad/tuple/group_depsNoOp+^critic/gradients/critic/MatMul_grad/MatMul-^critic/gradients/critic/MatMul_grad/MatMul_1

<critic/gradients/critic/MatMul_grad/tuple/control_dependencyIdentity*critic/gradients/critic/MatMul_grad/MatMul5^critic/gradients/critic/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@critic/gradients/critic/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

>critic/gradients/critic/MatMul_grad/tuple/control_dependency_1Identity,critic/gradients/critic/MatMul_grad/MatMul_15^critic/gradients/critic/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@critic/gradients/critic/MatMul_grad/MatMul_1*
_output_shapes
:	

 critic/beta1_power/initial_valueConst*
_class
loc:@critic/b1*
valueB
 *fff?*
dtype0*
_output_shapes
: 

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
Á
critic/beta1_power/AssignAssigncritic/beta1_power critic/beta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@critic/b1
v
critic/beta1_power/readIdentitycritic/beta1_power*
T0*
_class
loc:@critic/b1*
_output_shapes
: 

 critic/beta2_power/initial_valueConst*
_class
loc:@critic/b1*
valueB
 *wž?*
dtype0*
_output_shapes
: 

critic/beta2_power
VariableV2*
_class
loc:@critic/b1*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Á
critic/beta2_power/AssignAssigncritic/beta2_power critic/beta2_power/initial_value*
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
: *
use_locking(
v
critic/beta2_power/readIdentitycritic/beta2_power*
T0*
_class
loc:@critic/b1*
_output_shapes
: 
Ş
9critic/critic/w1_s/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@critic/w1_s*
valueB"      

/critic/critic/w1_s/Adam/Initializer/zeros/ConstConst*
_class
loc:@critic/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
ů
)critic/critic/w1_s/Adam/Initializer/zerosFill9critic/critic/w1_s/Adam/Initializer/zeros/shape_as_tensor/critic/critic/w1_s/Adam/Initializer/zeros/Const*
_output_shapes
:	*
T0*
_class
loc:@critic/w1_s*

index_type0
­
critic/critic/w1_s/Adam
VariableV2*
_class
loc:@critic/w1_s*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 
ß
critic/critic/w1_s/Adam/AssignAssigncritic/critic/w1_s/Adam)critic/critic/w1_s/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@critic/w1_s*
validate_shape(*
_output_shapes
:	

critic/critic/w1_s/Adam/readIdentitycritic/critic/w1_s/Adam*
T0*
_class
loc:@critic/w1_s*
_output_shapes
:	
Ź
;critic/critic/w1_s/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@critic/w1_s*
valueB"      *
dtype0*
_output_shapes
:

1critic/critic/w1_s/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@critic/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
˙
+critic/critic/w1_s/Adam_1/Initializer/zerosFill;critic/critic/w1_s/Adam_1/Initializer/zeros/shape_as_tensor1critic/critic/w1_s/Adam_1/Initializer/zeros/Const*
_output_shapes
:	*
T0*
_class
loc:@critic/w1_s*

index_type0
Ż
critic/critic/w1_s/Adam_1
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *
_class
loc:@critic/w1_s*
	container 
ĺ
 critic/critic/w1_s/Adam_1/AssignAssigncritic/critic/w1_s/Adam_1+critic/critic/w1_s/Adam_1/Initializer/zeros*
T0*
_class
loc:@critic/w1_s*
validate_shape(*
_output_shapes
:	*
use_locking(

critic/critic/w1_s/Adam_1/readIdentitycritic/critic/w1_s/Adam_1*
_class
loc:@critic/w1_s*
_output_shapes
:	*
T0

'critic/critic/b1/Adam/Initializer/zerosConst*
_class
loc:@critic/b1*
valueB	*    *
dtype0*
_output_shapes
:	
Š
critic/critic/b1/Adam
VariableV2*
shared_name *
_class
loc:@critic/b1*
	container *
shape:	*
dtype0*
_output_shapes
:	
×
critic/critic/b1/Adam/AssignAssigncritic/critic/b1/Adam'critic/critic/b1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
:	

critic/critic/b1/Adam/readIdentitycritic/critic/b1/Adam*
_output_shapes
:	*
T0*
_class
loc:@critic/b1

)critic/critic/b1/Adam_1/Initializer/zerosConst*
_class
loc:@critic/b1*
valueB	*    *
dtype0*
_output_shapes
:	
Ť
critic/critic/b1/Adam_1
VariableV2*
shared_name *
_class
loc:@critic/b1*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ý
critic/critic/b1/Adam_1/AssignAssigncritic/critic/b1/Adam_1)critic/critic/b1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@critic/b1

critic/critic/b1/Adam_1/readIdentitycritic/critic/b1/Adam_1*
T0*
_class
loc:@critic/b1*
_output_shapes
:	
´
>critic/critic/l2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*#
_class
loc:@critic/l2/kernel*
valueB"      *
dtype0

4critic/critic/l2/kernel/Adam/Initializer/zeros/ConstConst*#
_class
loc:@critic/l2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

.critic/critic/l2/kernel/Adam/Initializer/zerosFill>critic/critic/l2/kernel/Adam/Initializer/zeros/shape_as_tensor4critic/critic/l2/kernel/Adam/Initializer/zeros/Const*
T0*#
_class
loc:@critic/l2/kernel*

index_type0* 
_output_shapes
:

š
critic/critic/l2/kernel/Adam
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *#
_class
loc:@critic/l2/kernel
ô
#critic/critic/l2/kernel/Adam/AssignAssigncritic/critic/l2/kernel/Adam.critic/critic/l2/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@critic/l2/kernel

!critic/critic/l2/kernel/Adam/readIdentitycritic/critic/l2/kernel/Adam*
T0*#
_class
loc:@critic/l2/kernel* 
_output_shapes
:

ś
@critic/critic/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@critic/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:
 
6critic/critic/l2/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *#
_class
loc:@critic/l2/kernel*
valueB
 *    *
dtype0

0critic/critic/l2/kernel/Adam_1/Initializer/zerosFill@critic/critic/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensor6critic/critic/l2/kernel/Adam_1/Initializer/zeros/Const*#
_class
loc:@critic/l2/kernel*

index_type0* 
_output_shapes
:
*
T0
ť
critic/critic/l2/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *#
_class
loc:@critic/l2/kernel*
	container *
shape:

ú
%critic/critic/l2/kernel/Adam_1/AssignAssigncritic/critic/l2/kernel/Adam_10critic/critic/l2/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@critic/l2/kernel

#critic/critic/l2/kernel/Adam_1/readIdentitycritic/critic/l2/kernel/Adam_1*
T0*#
_class
loc:@critic/l2/kernel* 
_output_shapes
:


,critic/critic/l2/bias/Adam/Initializer/zerosConst*!
_class
loc:@critic/l2/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ť
critic/critic/l2/bias/Adam
VariableV2*
shared_name *!
_class
loc:@critic/l2/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ç
!critic/critic/l2/bias/Adam/AssignAssigncritic/critic/l2/bias/Adam,critic/critic/l2/bias/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@critic/l2/bias*
validate_shape(*
_output_shapes	
:

critic/critic/l2/bias/Adam/readIdentitycritic/critic/l2/bias/Adam*
T0*!
_class
loc:@critic/l2/bias*
_output_shapes	
:
 
.critic/critic/l2/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@critic/l2/bias*
valueB*    *
dtype0*
_output_shapes	
:
­
critic/critic/l2/bias/Adam_1
VariableV2*
shared_name *!
_class
loc:@critic/l2/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
í
#critic/critic/l2/bias/Adam_1/AssignAssigncritic/critic/l2/bias/Adam_1.critic/critic/l2/bias/Adam_1/Initializer/zeros*!
_class
loc:@critic/l2/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

!critic/critic/l2/bias/Adam_1/readIdentitycritic/critic/l2/bias/Adam_1*
T0*!
_class
loc:@critic/l2/bias*
_output_shapes	
:
´
>critic/critic/l3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@critic/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:

4critic/critic/l3/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *#
_class
loc:@critic/l3/kernel*
valueB
 *    

.critic/critic/l3/kernel/Adam/Initializer/zerosFill>critic/critic/l3/kernel/Adam/Initializer/zeros/shape_as_tensor4critic/critic/l3/kernel/Adam/Initializer/zeros/Const*
T0*#
_class
loc:@critic/l3/kernel*

index_type0* 
_output_shapes
:

š
critic/critic/l3/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *#
_class
loc:@critic/l3/kernel*
	container *
shape:

ô
#critic/critic/l3/kernel/Adam/AssignAssigncritic/critic/l3/kernel/Adam.critic/critic/l3/kernel/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@critic/l3/kernel*
validate_shape(* 
_output_shapes
:


!critic/critic/l3/kernel/Adam/readIdentitycritic/critic/l3/kernel/Adam*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:

ś
@critic/critic/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@critic/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:
 
6critic/critic/l3/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *#
_class
loc:@critic/l3/kernel*
valueB
 *    

0critic/critic/l3/kernel/Adam_1/Initializer/zerosFill@critic/critic/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensor6critic/critic/l3/kernel/Adam_1/Initializer/zeros/Const*
T0*#
_class
loc:@critic/l3/kernel*

index_type0* 
_output_shapes
:

ť
critic/critic/l3/kernel/Adam_1
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *#
_class
loc:@critic/l3/kernel
ú
%critic/critic/l3/kernel/Adam_1/AssignAssigncritic/critic/l3/kernel/Adam_10critic/critic/l3/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@critic/l3/kernel

#critic/critic/l3/kernel/Adam_1/readIdentitycritic/critic/l3/kernel/Adam_1*
T0*#
_class
loc:@critic/l3/kernel* 
_output_shapes
:


,critic/critic/l3/bias/Adam/Initializer/zerosConst*!
_class
loc:@critic/l3/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ť
critic/critic/l3/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *!
_class
loc:@critic/l3/bias*
	container 
ç
!critic/critic/l3/bias/Adam/AssignAssigncritic/critic/l3/bias/Adam,critic/critic/l3/bias/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@critic/l3/bias*
validate_shape(*
_output_shapes	
:

critic/critic/l3/bias/Adam/readIdentitycritic/critic/l3/bias/Adam*
_output_shapes	
:*
T0*!
_class
loc:@critic/l3/bias
 
.critic/critic/l3/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@critic/l3/bias*
valueB*    *
dtype0*
_output_shapes	
:
­
critic/critic/l3/bias/Adam_1
VariableV2*
_output_shapes	
:*
shared_name *!
_class
loc:@critic/l3/bias*
	container *
shape:*
dtype0
í
#critic/critic/l3/bias/Adam_1/AssignAssigncritic/critic/l3/bias/Adam_1.critic/critic/l3/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@critic/l3/bias*
validate_shape(*
_output_shapes	
:

!critic/critic/l3/bias/Adam_1/readIdentitycritic/critic/l3/bias/Adam_1*
_output_shapes	
:*
T0*!
_class
loc:@critic/l3/bias
°
1critic/critic/dense/kernel/Adam/Initializer/zerosConst*&
_class
loc:@critic/dense/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
˝
critic/critic/dense/kernel/Adam
VariableV2*&
_class
loc:@critic/dense/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 
˙
&critic/critic/dense/kernel/Adam/AssignAssigncritic/critic/dense/kernel/Adam1critic/critic/dense/kernel/Adam/Initializer/zeros*&
_class
loc:@critic/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
Ł
$critic/critic/dense/kernel/Adam/readIdentitycritic/critic/dense/kernel/Adam*
_output_shapes
:	*
T0*&
_class
loc:@critic/dense/kernel
˛
3critic/critic/dense/kernel/Adam_1/Initializer/zerosConst*
_output_shapes
:	*&
_class
loc:@critic/dense/kernel*
valueB	*    *
dtype0
ż
!critic/critic/dense/kernel/Adam_1
VariableV2*
shared_name *&
_class
loc:@critic/dense/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	

(critic/critic/dense/kernel/Adam_1/AssignAssign!critic/critic/dense/kernel/Adam_13critic/critic/dense/kernel/Adam_1/Initializer/zeros*
T0*&
_class
loc:@critic/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
§
&critic/critic/dense/kernel/Adam_1/readIdentity!critic/critic/dense/kernel/Adam_1*
T0*&
_class
loc:@critic/dense/kernel*
_output_shapes
:	
˘
/critic/critic/dense/bias/Adam/Initializer/zerosConst*$
_class
loc:@critic/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Ż
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
ň
$critic/critic/dense/bias/Adam/AssignAssigncritic/critic/dense/bias/Adam/critic/critic/dense/bias/Adam/Initializer/zeros*
T0*$
_class
loc:@critic/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(

"critic/critic/dense/bias/Adam/readIdentitycritic/critic/dense/bias/Adam*
T0*$
_class
loc:@critic/dense/bias*
_output_shapes
:
¤
1critic/critic/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*$
_class
loc:@critic/dense/bias*
valueB*    *
dtype0
ą
critic/critic/dense/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@critic/dense/bias
ř
&critic/critic/dense/bias/Adam_1/AssignAssigncritic/critic/dense/bias/Adam_11critic/critic/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@critic/dense/bias*
validate_shape(*
_output_shapes
:

$critic/critic/dense/bias/Adam_1/readIdentitycritic/critic/dense/bias/Adam_1*
_output_shapes
:*
T0*$
_class
loc:@critic/dense/bias
^
critic/Adam/learning_rateConst*
valueB
 *ˇQ9*
dtype0*
_output_shapes
: 
V
critic/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
V
critic/Adam/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
X
critic/Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
Ż
(critic/Adam/update_critic/w1_s/ApplyAdam	ApplyAdamcritic/w1_scritic/critic/w1_s/Adamcritic/critic/w1_s/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilon>critic/gradients/critic/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@critic/w1_s*
use_nesterov( *
_output_shapes
:	
˘
&critic/Adam/update_critic/b1/ApplyAdam	ApplyAdam	critic/b1critic/critic/b1/Adamcritic/critic/b1/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilon;critic/gradients/critic/add_grad/tuple/control_dependency_1*
_class
loc:@critic/b1*
use_nesterov( *
_output_shapes
:	*
use_locking( *
T0
Ě
-critic/Adam/update_critic/l2/kernel/ApplyAdam	ApplyAdamcritic/l2/kernelcritic/critic/l2/kernel/Adamcritic/critic/l2/kernel/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonAcritic/gradients/critic/l2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@critic/l2/kernel*
use_nesterov( * 
_output_shapes
:

ž
+critic/Adam/update_critic/l2/bias/ApplyAdam	ApplyAdamcritic/l2/biascritic/critic/l2/bias/Adamcritic/critic/l2/bias/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonBcritic/gradients/critic/l2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@critic/l2/bias*
use_nesterov( *
_output_shapes	
:
Ě
-critic/Adam/update_critic/l3/kernel/ApplyAdam	ApplyAdamcritic/l3/kernelcritic/critic/l3/kernel/Adamcritic/critic/l3/kernel/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonAcritic/gradients/critic/l3/MatMul_grad/tuple/control_dependency_1*#
_class
loc:@critic/l3/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
ž
+critic/Adam/update_critic/l3/bias/ApplyAdam	ApplyAdamcritic/l3/biascritic/critic/l3/bias/Adamcritic/critic/l3/bias/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonBcritic/gradients/critic/l3/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*!
_class
loc:@critic/l3/bias*
use_nesterov( 
Ý
0critic/Adam/update_critic/dense/kernel/ApplyAdam	ApplyAdamcritic/dense/kernelcritic/critic/dense/kernel/Adam!critic/critic/dense/kernel/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonDcritic/gradients/critic/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@critic/dense/kernel*
use_nesterov( *
_output_shapes
:	
Ď
.critic/Adam/update_critic/dense/bias/ApplyAdam	ApplyAdamcritic/dense/biascritic/critic/dense/bias/Adamcritic/critic/dense/bias/Adam_1critic/beta1_power/readcritic/beta2_power/readcritic/Adam/learning_ratecritic/Adam/beta1critic/Adam/beta2critic/Adam/epsilonEcritic/gradients/critic/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@critic/dense/bias*
use_nesterov( *
_output_shapes
:
ő
critic/Adam/mulMulcritic/beta1_power/readcritic/Adam/beta1'^critic/Adam/update_critic/b1/ApplyAdam/^critic/Adam/update_critic/dense/bias/ApplyAdam1^critic/Adam/update_critic/dense/kernel/ApplyAdam,^critic/Adam/update_critic/l2/bias/ApplyAdam.^critic/Adam/update_critic/l2/kernel/ApplyAdam,^critic/Adam/update_critic/l3/bias/ApplyAdam.^critic/Adam/update_critic/l3/kernel/ApplyAdam)^critic/Adam/update_critic/w1_s/ApplyAdam*
T0*
_class
loc:@critic/b1*
_output_shapes
: 
Š
critic/Adam/AssignAssigncritic/beta1_powercritic/Adam/mul*
use_locking( *
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
: 
÷
critic/Adam/mul_1Mulcritic/beta2_power/readcritic/Adam/beta2'^critic/Adam/update_critic/b1/ApplyAdam/^critic/Adam/update_critic/dense/bias/ApplyAdam1^critic/Adam/update_critic/dense/kernel/ApplyAdam,^critic/Adam/update_critic/l2/bias/ApplyAdam.^critic/Adam/update_critic/l2/kernel/ApplyAdam,^critic/Adam/update_critic/l3/bias/ApplyAdam.^critic/Adam/update_critic/l3/kernel/ApplyAdam)^critic/Adam/update_critic/w1_s/ApplyAdam*
T0*
_class
loc:@critic/b1*
_output_shapes
: 
­
critic/Adam/Assign_1Assigncritic/beta2_powercritic/Adam/mul_1*
use_locking( *
T0*
_class
loc:@critic/b1*
validate_shape(*
_output_shapes
: 
ł
critic/AdamNoOp^critic/Adam/Assign^critic/Adam/Assign_1'^critic/Adam/update_critic/b1/ApplyAdam/^critic/Adam/update_critic/dense/bias/ApplyAdam1^critic/Adam/update_critic/dense/kernel/ApplyAdam,^critic/Adam/update_critic/l2/bias/ApplyAdam.^critic/Adam/update_critic/l2/kernel/ApplyAdam,^critic/Adam/update_critic/l3/bias/ApplyAdam.^critic/Adam/update_critic/l3/kernel/ApplyAdam)^critic/Adam/update_critic/w1_s/ApplyAdam

-pi/l1/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/l1/kernel*
valueB"      *
dtype0*
_output_shapes
:

+pi/l1/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/l1/kernel*
valueB
 *°îž*
dtype0*
_output_shapes
: 

+pi/l1/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/l1/kernel*
valueB
 *°î>*
dtype0*
_output_shapes
: 
ć
5pi/l1/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l1/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*
_class
loc:@pi/l1/kernel*
seed2 
Î
+pi/l1/kernel/Initializer/random_uniform/subSub+pi/l1/kernel/Initializer/random_uniform/max+pi/l1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@pi/l1/kernel
á
+pi/l1/kernel/Initializer/random_uniform/mulMul5pi/l1/kernel/Initializer/random_uniform/RandomUniform+pi/l1/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@pi/l1/kernel*
_output_shapes
:	
Ó
'pi/l1/kernel/Initializer/random_uniformAdd+pi/l1/kernel/Initializer/random_uniform/mul+pi/l1/kernel/Initializer/random_uniform/min*
_output_shapes
:	*
T0*
_class
loc:@pi/l1/kernel
Ł
pi/l1/kernel
VariableV2*
_class
loc:@pi/l1/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 
Č
pi/l1/kernel/AssignAssignpi/l1/kernel'pi/l1/kernel/Initializer/random_uniform*
T0*
_class
loc:@pi/l1/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
v
pi/l1/kernel/readIdentitypi/l1/kernel*
T0*
_class
loc:@pi/l1/kernel*
_output_shapes
:	

pi/l1/bias/Initializer/zerosConst*
_class
loc:@pi/l1/bias*
valueB*    *
dtype0*
_output_shapes	
:


pi/l1/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l1/bias*
	container 
ł
pi/l1/bias/AssignAssign
pi/l1/biaspi/l1/bias/Initializer/zeros*
_class
loc:@pi/l1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
l
pi/l1/bias/readIdentity
pi/l1/bias*
T0*
_class
loc:@pi/l1/bias*
_output_shapes	
:

pi/l1/MatMulMatMulstatepi/l1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

pi/l1/BiasAddBiasAddpi/l1/MatMulpi/l1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
T

pi/l1/ReluRelupi/l1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

-pi/l2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
_class
loc:@pi/l2/kernel*
valueB"      *
dtype0

+pi/l2/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/l2/kernel*
valueB
 *×łÝ˝*
dtype0*
_output_shapes
: 

+pi/l2/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/l2/kernel*
valueB
 *×łÝ=*
dtype0*
_output_shapes
: 
ç
5pi/l2/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l2/kernel/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@pi/l2/kernel*
seed2 *
dtype0* 
_output_shapes
:

Î
+pi/l2/kernel/Initializer/random_uniform/subSub+pi/l2/kernel/Initializer/random_uniform/max+pi/l2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@pi/l2/kernel
â
+pi/l2/kernel/Initializer/random_uniform/mulMul5pi/l2/kernel/Initializer/random_uniform/RandomUniform+pi/l2/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@pi/l2/kernel* 
_output_shapes
:

Ô
'pi/l2/kernel/Initializer/random_uniformAdd+pi/l2/kernel/Initializer/random_uniform/mul+pi/l2/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l2/kernel* 
_output_shapes
:

Ľ
pi/l2/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *
_class
loc:@pi/l2/kernel
É
pi/l2/kernel/AssignAssignpi/l2/kernel'pi/l2/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@pi/l2/kernel*
validate_shape(* 
_output_shapes
:

w
pi/l2/kernel/readIdentitypi/l2/kernel* 
_output_shapes
:
*
T0*
_class
loc:@pi/l2/kernel

pi/l2/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
_class
loc:@pi/l2/bias*
valueB*    


pi/l2/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l2/bias*
	container *
shape:
ł
pi/l2/bias/AssignAssign
pi/l2/biaspi/l2/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
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
:

pi/l2/MatMulMatMul
pi/l1/Relupi/l2/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

pi/l2/BiasAddBiasAddpi/l2/MatMulpi/l2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
T

pi/l2/ReluRelupi/l2/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-pi/l3/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:

+pi/l3/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/l3/kernel*
valueB
 *×łÝ˝*
dtype0*
_output_shapes
: 

+pi/l3/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
_class
loc:@pi/l3/kernel*
valueB
 *×łÝ=*
dtype0
ç
5pi/l3/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l3/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*
_class
loc:@pi/l3/kernel
Î
+pi/l3/kernel/Initializer/random_uniform/subSub+pi/l3/kernel/Initializer/random_uniform/max+pi/l3/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l3/kernel*
_output_shapes
: 
â
+pi/l3/kernel/Initializer/random_uniform/mulMul5pi/l3/kernel/Initializer/random_uniform/RandomUniform+pi/l3/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*
_class
loc:@pi/l3/kernel
Ô
'pi/l3/kernel/Initializer/random_uniformAdd+pi/l3/kernel/Initializer/random_uniform/mul+pi/l3/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*
_class
loc:@pi/l3/kernel
Ľ
pi/l3/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *
_class
loc:@pi/l3/kernel
É
pi/l3/kernel/AssignAssignpi/l3/kernel'pi/l3/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@pi/l3/kernel*
validate_shape(* 
_output_shapes
:

w
pi/l3/kernel/readIdentitypi/l3/kernel*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:


pi/l3/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
_class
loc:@pi/l3/bias*
valueB*    


pi/l3/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l3/bias*
	container *
shape:
ł
pi/l3/bias/AssignAssign
pi/l3/biaspi/l3/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l3/bias*
validate_shape(*
_output_shapes	
:
l
pi/l3/bias/readIdentity
pi/l3/bias*
T0*
_class
loc:@pi/l3/bias*
_output_shapes	
:

pi/l3/MatMulMatMul
pi/l2/Relupi/l3/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

pi/l3/BiasAddBiasAddpi/l3/MatMulpi/l3/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
T

pi/l3/ReluRelupi/l3/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-pi/l4/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/l4/kernel*
valueB"      *
dtype0*
_output_shapes
:

+pi/l4/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
_class
loc:@pi/l4/kernel*
valueB
 *   ž*
dtype0

+pi/l4/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
_class
loc:@pi/l4/kernel*
valueB
 *   >
ç
5pi/l4/kernel/Initializer/random_uniform/RandomUniformRandomUniform-pi/l4/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*
_class
loc:@pi/l4/kernel*
seed2 
Î
+pi/l4/kernel/Initializer/random_uniform/subSub+pi/l4/kernel/Initializer/random_uniform/max+pi/l4/kernel/Initializer/random_uniform/min*
_class
loc:@pi/l4/kernel*
_output_shapes
: *
T0
â
+pi/l4/kernel/Initializer/random_uniform/mulMul5pi/l4/kernel/Initializer/random_uniform/RandomUniform+pi/l4/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@pi/l4/kernel* 
_output_shapes
:

Ô
'pi/l4/kernel/Initializer/random_uniformAdd+pi/l4/kernel/Initializer/random_uniform/mul+pi/l4/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/l4/kernel* 
_output_shapes
:

Ľ
pi/l4/kernel
VariableV2*
shared_name *
_class
loc:@pi/l4/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

É
pi/l4/kernel/AssignAssignpi/l4/kernel'pi/l4/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@pi/l4/kernel*
validate_shape(* 
_output_shapes
:

w
pi/l4/kernel/readIdentitypi/l4/kernel*
_class
loc:@pi/l4/kernel* 
_output_shapes
:
*
T0

pi/l4/bias/Initializer/zerosConst*
_class
loc:@pi/l4/bias*
valueB*    *
dtype0*
_output_shapes	
:


pi/l4/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l4/bias*
	container *
shape:
ł
pi/l4/bias/AssignAssign
pi/l4/biaspi/l4/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l4/bias*
validate_shape(*
_output_shapes	
:
l
pi/l4/bias/readIdentity
pi/l4/bias*
T0*
_class
loc:@pi/l4/bias*
_output_shapes	
:

pi/l4/MatMulMatMul
pi/l3/Relupi/l4/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

pi/l4/BiasAddBiasAddpi/l4/MatMulpi/l4/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
T

pi/l4/ReluRelupi/l4/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

,pi/a/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@pi/a/kernel*
valueB"      *
dtype0*
_output_shapes
:

*pi/a/kernel/Initializer/random_uniform/minConst*
_class
loc:@pi/a/kernel*
valueB
 *JQZž*
dtype0*
_output_shapes
: 

*pi/a/kernel/Initializer/random_uniform/maxConst*
_class
loc:@pi/a/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
ă
4pi/a/kernel/Initializer/random_uniform/RandomUniformRandomUniform,pi/a/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*
_class
loc:@pi/a/kernel*
seed2 
Ę
*pi/a/kernel/Initializer/random_uniform/subSub*pi/a/kernel/Initializer/random_uniform/max*pi/a/kernel/Initializer/random_uniform/min*
_class
loc:@pi/a/kernel*
_output_shapes
: *
T0
Ý
*pi/a/kernel/Initializer/random_uniform/mulMul4pi/a/kernel/Initializer/random_uniform/RandomUniform*pi/a/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*
_class
loc:@pi/a/kernel
Ď
&pi/a/kernel/Initializer/random_uniformAdd*pi/a/kernel/Initializer/random_uniform/mul*pi/a/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@pi/a/kernel*
_output_shapes
:	
Ą
pi/a/kernel
VariableV2*
shared_name *
_class
loc:@pi/a/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ä
pi/a/kernel/AssignAssignpi/a/kernel&pi/a/kernel/Initializer/random_uniform*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@pi/a/kernel*
validate_shape(
s
pi/a/kernel/readIdentitypi/a/kernel*
T0*
_class
loc:@pi/a/kernel*
_output_shapes
:	

pi/a/bias/Initializer/zerosConst*
_class
loc:@pi/a/bias*
valueB*    *
dtype0*
_output_shapes
:

	pi/a/bias
VariableV2*
_class
loc:@pi/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ž
pi/a/bias/AssignAssign	pi/a/biaspi/a/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
:
h
pi/a/bias/readIdentity	pi/a/bias*
_class
loc:@pi/a/bias*
_output_shapes
:*
T0

pi/a/MatMulMatMul
pi/l4/Relupi/a/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
}
pi/a/BiasAddBiasAddpi/a/MatMulpi/a/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
	pi/a/TanhTanhpi/a/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
g
pi/scaled_mu/yConst*%
valueB"Z<?ˇŃ8˝75ˇŃ8*
dtype0*
_output_shapes
:
`
pi/scaled_muMul	pi/a/Tanhpi/scaled_mu/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0pi/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*"
_class
loc:@pi/dense/kernel*
valueB"      *
dtype0

.pi/dense/kernel/Initializer/random_uniform/minConst*"
_class
loc:@pi/dense/kernel*
valueB
 *JQZž*
dtype0*
_output_shapes
: 

.pi/dense/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@pi/dense/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
ď
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	*

seed *
T0*"
_class
loc:@pi/dense/kernel
Ú
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@pi/dense/kernel*
_output_shapes
: *
T0
í
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	
ß
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	*
T0*"
_class
loc:@pi/dense/kernel
Š
pi/dense/kernel
VariableV2*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ô
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	

pi/dense/kernel/readIdentitypi/dense/kernel*
_output_shapes
:	*
T0*"
_class
loc:@pi/dense/kernel

pi/dense/bias/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0*
_output_shapes
:

pi/dense/bias
VariableV2*
shared_name * 
_class
loc:@pi/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ž
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:
t
pi/dense/bias/readIdentitypi/dense/bias*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:

pi/dense/MatMulMatMul
pi/l4/Relupi/dense/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
pi/dense/SoftplusSoftpluspi/dense/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
pi/scaled_sigma/yConst*%
valueB"Z<?ˇŃ8˝75ˇŃ8*
dtype0*
_output_shapes
:
n
pi/scaled_sigmaMulpi/dense/Softpluspi/scaled_sigma/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
pi/Normal/locIdentitypi/scaled_mu*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
pi/Normal/scaleIdentitypi/scaled_sigma*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0oldpi/l1/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@oldpi/l1/kernel*
valueB"      *
dtype0*
_output_shapes
:

.oldpi/l1/kernel/Initializer/random_uniform/minConst*"
_class
loc:@oldpi/l1/kernel*
valueB
 *°îž*
dtype0*
_output_shapes
: 

.oldpi/l1/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@oldpi/l1/kernel*
valueB
 *°î>*
dtype0*
_output_shapes
: 
ď
8oldpi/l1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l1/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*"
_class
loc:@oldpi/l1/kernel*
seed2 
Ú
.oldpi/l1/kernel/Initializer/random_uniform/subSub.oldpi/l1/kernel/Initializer/random_uniform/max.oldpi/l1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@oldpi/l1/kernel
í
.oldpi/l1/kernel/Initializer/random_uniform/mulMul8oldpi/l1/kernel/Initializer/random_uniform/RandomUniform.oldpi/l1/kernel/Initializer/random_uniform/sub*"
_class
loc:@oldpi/l1/kernel*
_output_shapes
:	*
T0
ß
*oldpi/l1/kernel/Initializer/random_uniformAdd.oldpi/l1/kernel/Initializer/random_uniform/mul.oldpi/l1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l1/kernel*
_output_shapes
:	
Š
oldpi/l1/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *"
_class
loc:@oldpi/l1/kernel*
	container *
shape:	
Ô
oldpi/l1/kernel/AssignAssignoldpi/l1/kernel*oldpi/l1/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*"
_class
loc:@oldpi/l1/kernel

oldpi/l1/kernel/readIdentityoldpi/l1/kernel*"
_class
loc:@oldpi/l1/kernel*
_output_shapes
:	*
T0

oldpi/l1/bias/Initializer/zerosConst* 
_class
loc:@oldpi/l1/bias*
valueB*    *
dtype0*
_output_shapes	
:

oldpi/l1/bias
VariableV2* 
_class
loc:@oldpi/l1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ż
oldpi/l1/bias/AssignAssignoldpi/l1/biasoldpi/l1/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@oldpi/l1/bias*
validate_shape(
u
oldpi/l1/bias/readIdentityoldpi/l1/bias*
_output_shapes	
:*
T0* 
_class
loc:@oldpi/l1/bias

oldpi/l1/MatMulMatMulstateoldpi/l1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

oldpi/l1/BiasAddBiasAddoldpi/l1/MatMuloldpi/l1/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Z
oldpi/l1/ReluReluoldpi/l1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0oldpi/l2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@oldpi/l2/kernel*
valueB"      

.oldpi/l2/kernel/Initializer/random_uniform/minConst*"
_class
loc:@oldpi/l2/kernel*
valueB
 *×łÝ˝*
dtype0*
_output_shapes
: 

.oldpi/l2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@oldpi/l2/kernel*
valueB
 *×łÝ=
đ
8oldpi/l2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l2/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*"
_class
loc:@oldpi/l2/kernel*
seed2 
Ú
.oldpi/l2/kernel/Initializer/random_uniform/subSub.oldpi/l2/kernel/Initializer/random_uniform/max.oldpi/l2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l2/kernel*
_output_shapes
: 
î
.oldpi/l2/kernel/Initializer/random_uniform/mulMul8oldpi/l2/kernel/Initializer/random_uniform/RandomUniform.oldpi/l2/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@oldpi/l2/kernel* 
_output_shapes
:

ŕ
*oldpi/l2/kernel/Initializer/random_uniformAdd.oldpi/l2/kernel/Initializer/random_uniform/mul.oldpi/l2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l2/kernel* 
_output_shapes
:

Ť
oldpi/l2/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *"
_class
loc:@oldpi/l2/kernel*
	container *
shape:

Ő
oldpi/l2/kernel/AssignAssignoldpi/l2/kernel*oldpi/l2/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@oldpi/l2/kernel*
validate_shape(* 
_output_shapes
:


oldpi/l2/kernel/readIdentityoldpi/l2/kernel*
T0*"
_class
loc:@oldpi/l2/kernel* 
_output_shapes
:


oldpi/l2/bias/Initializer/zerosConst* 
_class
loc:@oldpi/l2/bias*
valueB*    *
dtype0*
_output_shapes	
:

oldpi/l2/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name * 
_class
loc:@oldpi/l2/bias*
	container 
ż
oldpi/l2/bias/AssignAssignoldpi/l2/biasoldpi/l2/bias/Initializer/zeros*
T0* 
_class
loc:@oldpi/l2/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
u
oldpi/l2/bias/readIdentityoldpi/l2/bias*
_output_shapes	
:*
T0* 
_class
loc:@oldpi/l2/bias

oldpi/l2/MatMulMatMuloldpi/l1/Reluoldpi/l2/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

oldpi/l2/BiasAddBiasAddoldpi/l2/MatMuloldpi/l2/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
Z
oldpi/l2/ReluReluoldpi/l2/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0oldpi/l3/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@oldpi/l3/kernel*
valueB"      *
dtype0*
_output_shapes
:

.oldpi/l3/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@oldpi/l3/kernel*
valueB
 *×łÝ˝

.oldpi/l3/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@oldpi/l3/kernel*
valueB
 *×łÝ=*
dtype0
đ
8oldpi/l3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l3/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*"
_class
loc:@oldpi/l3/kernel
Ú
.oldpi/l3/kernel/Initializer/random_uniform/subSub.oldpi/l3/kernel/Initializer/random_uniform/max.oldpi/l3/kernel/Initializer/random_uniform/min*"
_class
loc:@oldpi/l3/kernel*
_output_shapes
: *
T0
î
.oldpi/l3/kernel/Initializer/random_uniform/mulMul8oldpi/l3/kernel/Initializer/random_uniform/RandomUniform.oldpi/l3/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@oldpi/l3/kernel* 
_output_shapes
:

ŕ
*oldpi/l3/kernel/Initializer/random_uniformAdd.oldpi/l3/kernel/Initializer/random_uniform/mul.oldpi/l3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l3/kernel* 
_output_shapes
:

Ť
oldpi/l3/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *"
_class
loc:@oldpi/l3/kernel*
	container 
Ő
oldpi/l3/kernel/AssignAssignoldpi/l3/kernel*oldpi/l3/kernel/Initializer/random_uniform*
T0*"
_class
loc:@oldpi/l3/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

oldpi/l3/kernel/readIdentityoldpi/l3/kernel* 
_output_shapes
:
*
T0*"
_class
loc:@oldpi/l3/kernel

oldpi/l3/bias/Initializer/zerosConst* 
_class
loc:@oldpi/l3/bias*
valueB*    *
dtype0*
_output_shapes	
:

oldpi/l3/bias
VariableV2*
shared_name * 
_class
loc:@oldpi/l3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ż
oldpi/l3/bias/AssignAssignoldpi/l3/biasoldpi/l3/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@oldpi/l3/bias
u
oldpi/l3/bias/readIdentityoldpi/l3/bias* 
_class
loc:@oldpi/l3/bias*
_output_shapes	
:*
T0

oldpi/l3/MatMulMatMuloldpi/l2/Reluoldpi/l3/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

oldpi/l3/BiasAddBiasAddoldpi/l3/MatMuloldpi/l3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
oldpi/l3/ReluReluoldpi/l3/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
0oldpi/l4/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@oldpi/l4/kernel*
valueB"      *
dtype0*
_output_shapes
:

.oldpi/l4/kernel/Initializer/random_uniform/minConst*"
_class
loc:@oldpi/l4/kernel*
valueB
 *   ž*
dtype0*
_output_shapes
: 

.oldpi/l4/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@oldpi/l4/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
đ
8oldpi/l4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0oldpi/l4/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*"
_class
loc:@oldpi/l4/kernel*
seed2 
Ú
.oldpi/l4/kernel/Initializer/random_uniform/subSub.oldpi/l4/kernel/Initializer/random_uniform/max.oldpi/l4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@oldpi/l4/kernel*
_output_shapes
: 
î
.oldpi/l4/kernel/Initializer/random_uniform/mulMul8oldpi/l4/kernel/Initializer/random_uniform/RandomUniform.oldpi/l4/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@oldpi/l4/kernel* 
_output_shapes
:

ŕ
*oldpi/l4/kernel/Initializer/random_uniformAdd.oldpi/l4/kernel/Initializer/random_uniform/mul.oldpi/l4/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*"
_class
loc:@oldpi/l4/kernel
Ť
oldpi/l4/kernel
VariableV2*"
_class
loc:@oldpi/l4/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Ő
oldpi/l4/kernel/AssignAssignoldpi/l4/kernel*oldpi/l4/kernel/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*"
_class
loc:@oldpi/l4/kernel*
validate_shape(

oldpi/l4/kernel/readIdentityoldpi/l4/kernel*
T0*"
_class
loc:@oldpi/l4/kernel* 
_output_shapes
:


oldpi/l4/bias/Initializer/zerosConst* 
_class
loc:@oldpi/l4/bias*
valueB*    *
dtype0*
_output_shapes	
:

oldpi/l4/bias
VariableV2*
shared_name * 
_class
loc:@oldpi/l4/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ż
oldpi/l4/bias/AssignAssignoldpi/l4/biasoldpi/l4/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@oldpi/l4/bias*
validate_shape(*
_output_shapes	
:
u
oldpi/l4/bias/readIdentityoldpi/l4/bias*
T0* 
_class
loc:@oldpi/l4/bias*
_output_shapes	
:

oldpi/l4/MatMulMatMuloldpi/l3/Reluoldpi/l4/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

oldpi/l4/BiasAddBiasAddoldpi/l4/MatMuloldpi/l4/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
oldpi/l4/ReluReluoldpi/l4/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/oldpi/a/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*!
_class
loc:@oldpi/a/kernel*
valueB"      

-oldpi/a/kernel/Initializer/random_uniform/minConst*!
_class
loc:@oldpi/a/kernel*
valueB
 *JQZž*
dtype0*
_output_shapes
: 

-oldpi/a/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@oldpi/a/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
ě
7oldpi/a/kernel/Initializer/random_uniform/RandomUniformRandomUniform/oldpi/a/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	*

seed *
T0*!
_class
loc:@oldpi/a/kernel
Ö
-oldpi/a/kernel/Initializer/random_uniform/subSub-oldpi/a/kernel/Initializer/random_uniform/max-oldpi/a/kernel/Initializer/random_uniform/min*!
_class
loc:@oldpi/a/kernel*
_output_shapes
: *
T0
é
-oldpi/a/kernel/Initializer/random_uniform/mulMul7oldpi/a/kernel/Initializer/random_uniform/RandomUniform-oldpi/a/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*!
_class
loc:@oldpi/a/kernel
Ű
)oldpi/a/kernel/Initializer/random_uniformAdd-oldpi/a/kernel/Initializer/random_uniform/mul-oldpi/a/kernel/Initializer/random_uniform/min*!
_class
loc:@oldpi/a/kernel*
_output_shapes
:	*
T0
§
oldpi/a/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *!
_class
loc:@oldpi/a/kernel*
	container *
shape:	
Đ
oldpi/a/kernel/AssignAssignoldpi/a/kernel)oldpi/a/kernel/Initializer/random_uniform*
use_locking(*
T0*!
_class
loc:@oldpi/a/kernel*
validate_shape(*
_output_shapes
:	
|
oldpi/a/kernel/readIdentityoldpi/a/kernel*
T0*!
_class
loc:@oldpi/a/kernel*
_output_shapes
:	

oldpi/a/bias/Initializer/zerosConst*
_class
loc:@oldpi/a/bias*
valueB*    *
dtype0*
_output_shapes
:

oldpi/a/bias
VariableV2*
_output_shapes
:*
shared_name *
_class
loc:@oldpi/a/bias*
	container *
shape:*
dtype0
ş
oldpi/a/bias/AssignAssignoldpi/a/biasoldpi/a/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@oldpi/a/bias
q
oldpi/a/bias/readIdentityoldpi/a/bias*
T0*
_class
loc:@oldpi/a/bias*
_output_shapes
:

oldpi/a/MatMulMatMuloldpi/l4/Reluoldpi/a/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

oldpi/a/BiasAddBiasAddoldpi/a/MatMuloldpi/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
oldpi/a/TanhTanholdpi/a/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
oldpi/scaled_mu/yConst*%
valueB"Z<?ˇŃ8˝75ˇŃ8*
dtype0*
_output_shapes
:
i
oldpi/scaled_muMuloldpi/a/Tanholdpi/scaled_mu/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
3oldpi/dense/kernel/Initializer/random_uniform/shapeConst*%
_class
loc:@oldpi/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:

1oldpi/dense/kernel/Initializer/random_uniform/minConst*%
_class
loc:@oldpi/dense/kernel*
valueB
 *JQZž*
dtype0*
_output_shapes
: 

1oldpi/dense/kernel/Initializer/random_uniform/maxConst*%
_class
loc:@oldpi/dense/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
ř
;oldpi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform3oldpi/dense/kernel/Initializer/random_uniform/shape*

seed *
T0*%
_class
loc:@oldpi/dense/kernel*
seed2 *
dtype0*
_output_shapes
:	
ć
1oldpi/dense/kernel/Initializer/random_uniform/subSub1oldpi/dense/kernel/Initializer/random_uniform/max1oldpi/dense/kernel/Initializer/random_uniform/min*
T0*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
: 
ů
1oldpi/dense/kernel/Initializer/random_uniform/mulMul;oldpi/dense/kernel/Initializer/random_uniform/RandomUniform1oldpi/dense/kernel/Initializer/random_uniform/sub*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
:	*
T0
ë
-oldpi/dense/kernel/Initializer/random_uniformAdd1oldpi/dense/kernel/Initializer/random_uniform/mul1oldpi/dense/kernel/Initializer/random_uniform/min*
T0*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
:	
Ż
oldpi/dense/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *%
_class
loc:@oldpi/dense/kernel*
	container *
shape:	
ŕ
oldpi/dense/kernel/AssignAssignoldpi/dense/kernel-oldpi/dense/kernel/Initializer/random_uniform*
T0*%
_class
loc:@oldpi/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(

oldpi/dense/kernel/readIdentityoldpi/dense/kernel*
T0*%
_class
loc:@oldpi/dense/kernel*
_output_shapes
:	

"oldpi/dense/bias/Initializer/zerosConst*#
_class
loc:@oldpi/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Ą
oldpi/dense/bias
VariableV2*
_output_shapes
:*
shared_name *#
_class
loc:@oldpi/dense/bias*
	container *
shape:*
dtype0
Ę
oldpi/dense/bias/AssignAssignoldpi/dense/bias"oldpi/dense/bias/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@oldpi/dense/bias*
validate_shape(*
_output_shapes
:
}
oldpi/dense/bias/readIdentityoldpi/dense/bias*
T0*#
_class
loc:@oldpi/dense/bias*
_output_shapes
:

oldpi/dense/MatMulMatMuloldpi/l4/Reluoldpi/dense/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

oldpi/dense/BiasAddBiasAddoldpi/dense/MatMuloldpi/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
oldpi/dense/SoftplusSoftplusoldpi/dense/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
oldpi/scaled_sigma/yConst*
_output_shapes
:*%
valueB"Z<?ˇŃ8˝75ˇŃ8*
dtype0
w
oldpi/scaled_sigmaMuloldpi/dense/Softplusoldpi/scaled_sigma/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
oldpi/Normal/locIdentityoldpi/scaled_mu*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
oldpi/Normal/scaleIdentityoldpi/scaled_sigma*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
pi/Normal/sample/sample_shapeConst*
dtype0*
_output_shapes
: *
value	B :
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
Ş
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
É
pi/Normal/sample/concatConcatV2 pi/Normal/sample/concat/values_0*pi/Normal/batch_shape_tensor/BroadcastArgspi/Normal/sample/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
h
#pi/Normal/sample/random_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
j
%pi/Normal/sample/random_normal/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
É
3pi/Normal/sample/random_normal/RandomStandardNormalRandomStandardNormalpi/Normal/sample/concat*

seed *
T0*
dtype0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
seed2 
Ä
"pi/Normal/sample/random_normal/mulMul3pi/Normal/sample/random_normal/RandomStandardNormal%pi/Normal/sample/random_normal/stddev*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
­
pi/Normal/sample/random_normalAdd"pi/Normal/sample/random_normal/mul#pi/Normal/sample/random_normal/mean*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

pi/Normal/sample/mulMulpi/Normal/sample/random_normalpi/Normal/scale*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
pi/Normal/sample/addAddpi/Normal/sample/mulpi/Normal/loc*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
pi/Normal/sample/ShapeShapepi/Normal/sample/add*
_output_shapes
:*
T0*
out_type0
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
Ň
pi/Normal/sample/strided_sliceStridedSlicepi/Normal/sample/Shape$pi/Normal/sample/strided_slice/stack&pi/Normal/sample/strided_slice/stack_1&pi/Normal/sample/strided_slice/stack_2*
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask 
`
pi/Normal/sample/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ŕ
pi/Normal/sample/concat_1ConcatV2pi/Normal/sample/sample_shape_1pi/Normal/sample/strided_slicepi/Normal/sample/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N

pi/Normal/sample/ReshapeReshapepi/Normal/sample/addpi/Normal/sample/concat_1*
Tshape0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

sample_action/SqueezeSqueezepi/Normal/sample/Reshape*
squeeze_dims
 *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
update_oldpi/AssignAssignoldpi/l1/kernelpi/l1/kernel/read*
T0*"
_class
loc:@oldpi/l1/kernel*
validate_shape(*
_output_shapes
:	*
use_locking( 
°
update_oldpi/Assign_1Assignoldpi/l1/biaspi/l1/bias/read*
validate_shape(*
_output_shapes	
:*
use_locking( *
T0* 
_class
loc:@oldpi/l1/bias
ť
update_oldpi/Assign_2Assignoldpi/l2/kernelpi/l2/kernel/read*
use_locking( *
T0*"
_class
loc:@oldpi/l2/kernel*
validate_shape(* 
_output_shapes
:

°
update_oldpi/Assign_3Assignoldpi/l2/biaspi/l2/bias/read* 
_class
loc:@oldpi/l2/bias*
validate_shape(*
_output_shapes	
:*
use_locking( *
T0
ť
update_oldpi/Assign_4Assignoldpi/l3/kernelpi/l3/kernel/read*"
_class
loc:@oldpi/l3/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking( *
T0
°
update_oldpi/Assign_5Assignoldpi/l3/biaspi/l3/bias/read*
use_locking( *
T0* 
_class
loc:@oldpi/l3/bias*
validate_shape(*
_output_shapes	
:
ť
update_oldpi/Assign_6Assignoldpi/l4/kernelpi/l4/kernel/read*
validate_shape(* 
_output_shapes
:
*
use_locking( *
T0*"
_class
loc:@oldpi/l4/kernel
°
update_oldpi/Assign_7Assignoldpi/l4/biaspi/l4/bias/read* 
_class
loc:@oldpi/l4/bias*
validate_shape(*
_output_shapes	
:*
use_locking( *
T0
ˇ
update_oldpi/Assign_8Assignoldpi/a/kernelpi/a/kernel/read*
use_locking( *
T0*!
_class
loc:@oldpi/a/kernel*
validate_shape(*
_output_shapes
:	
Ź
update_oldpi/Assign_9Assignoldpi/a/biaspi/a/bias/read*
_class
loc:@oldpi/a/bias*
validate_shape(*
_output_shapes
:*
use_locking( *
T0
Ä
update_oldpi/Assign_10Assignoldpi/dense/kernelpi/dense/kernel/read*
_output_shapes
:	*
use_locking( *
T0*%
_class
loc:@oldpi/dense/kernel*
validate_shape(
š
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
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
	advantagePlaceholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
n
pi/Normal/prob/standardize/subSubactionpi/Normal/loc*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"pi/Normal/prob/standardize/truedivRealDivpi/Normal/prob/standardize/subpi/Normal/scale*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
pi/Normal/prob/SquareSquare"pi/Normal/prob/standardize/truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Y
pi/Normal/prob/mul/xConst*
valueB
 *   ż*
dtype0*
_output_shapes
: 
x
pi/Normal/prob/mulMulpi/Normal/prob/mul/xpi/Normal/prob/Square*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
pi/Normal/prob/LogLogpi/Normal/scale*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
pi/Normal/prob/add/xConst*
valueB
 *?k?*
dtype0*
_output_shapes
: 
u
pi/Normal/prob/addAddpi/Normal/prob/add/xpi/Normal/prob/Log*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
pi/Normal/prob/subSubpi/Normal/prob/mulpi/Normal/prob/add*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
_
pi/Normal/prob/ExpExppi/Normal/prob/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
!oldpi/Normal/prob/standardize/subSubactionoldpi/Normal/loc*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

%oldpi/Normal/prob/standardize/truedivRealDiv!oldpi/Normal/prob/standardize/suboldpi/Normal/scale*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
{
oldpi/Normal/prob/SquareSquare%oldpi/Normal/prob/standardize/truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
\
oldpi/Normal/prob/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *   ż

oldpi/Normal/prob/mulMuloldpi/Normal/prob/mul/xoldpi/Normal/prob/Square*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
oldpi/Normal/prob/LogLogoldpi/Normal/scale*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
oldpi/Normal/prob/add/xConst*
_output_shapes
: *
valueB
 *?k?*
dtype0
~
oldpi/Normal/prob/addAddoldpi/Normal/prob/add/xoldpi/Normal/prob/Log*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
oldpi/Normal/prob/subSuboldpi/Normal/prob/muloldpi/Normal/prob/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
oldpi/Normal/prob/ExpExpoldpi/Normal/prob/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
loss/surrogate/truedivRealDivpi/Normal/prob/Expoldpi/Normal/prob/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
loss/surrogate/mulMulloss/surrogate/truediv	advantage*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
loss/clip_by_value/Minimum/yConst*
valueB
 *?*
dtype0*
_output_shapes
: 

loss/clip_by_value/MinimumMinimumloss/surrogate/truedivloss/clip_by_value/Minimum/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
loss/clip_by_value/yConst*
valueB
 *ÍĚL?*
dtype0*
_output_shapes
: 

loss/clip_by_valueMaximumloss/clip_by_value/Minimumloss/clip_by_value/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
loss/mulMulloss/clip_by_value	advantage*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
loss/MinimumMinimumloss/surrogate/mulloss/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[

loss/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
i
	loss/MeanMeanloss/Minimum
loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
 *  ?*
dtype0*
_output_shapes
: 

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
ź
'atrain/gradients/loss/Mean_grad/ReshapeReshape"atrain/gradients/loss/Neg_grad/Neg-atrain/gradients/loss/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
q
%atrain/gradients/loss/Mean_grad/ShapeShapeloss/Minimum*
_output_shapes
:*
T0*
out_type0
Ŕ
$atrain/gradients/loss/Mean_grad/TileTile'atrain/gradients/loss/Mean_grad/Reshape%atrain/gradients/loss/Mean_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0
s
'atrain/gradients/loss/Mean_grad/Shape_1Shapeloss/Minimum*
out_type0*
_output_shapes
:*
T0
j
'atrain/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
o
%atrain/gradients/loss/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
ş
$atrain/gradients/loss/Mean_grad/ProdProd'atrain/gradients/loss/Mean_grad/Shape_1%atrain/gradients/loss/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
q
'atrain/gradients/loss/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
ž
&atrain/gradients/loss/Mean_grad/Prod_1Prod'atrain/gradients/loss/Mean_grad/Shape_2'atrain/gradients/loss/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
k
)atrain/gradients/loss/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
Ś
'atrain/gradients/loss/Mean_grad/MaximumMaximum&atrain/gradients/loss/Mean_grad/Prod_1)atrain/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
¤
(atrain/gradients/loss/Mean_grad/floordivFloorDiv$atrain/gradients/loss/Mean_grad/Prod'atrain/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 

$atrain/gradients/loss/Mean_grad/CastCast(atrain/gradients/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
°
'atrain/gradients/loss/Mean_grad/truedivRealDiv$atrain/gradients/loss/Mean_grad/Tile$atrain/gradients/loss/Mean_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
(atrain/gradients/loss/Minimum_grad/ShapeShapeloss/surrogate/mul*
T0*
out_type0*
_output_shapes
:
r
*atrain/gradients/loss/Minimum_grad/Shape_1Shapeloss/mul*
_output_shapes
:*
T0*
out_type0

*atrain/gradients/loss/Minimum_grad/Shape_2Shape'atrain/gradients/loss/Mean_grad/truediv*
_output_shapes
:*
T0*
out_type0
s
.atrain/gradients/loss/Minimum_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Đ
(atrain/gradients/loss/Minimum_grad/zerosFill*atrain/gradients/loss/Minimum_grad/Shape_2.atrain/gradients/loss/Minimum_grad/zeros/Const*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

,atrain/gradients/loss/Minimum_grad/LessEqual	LessEqualloss/surrogate/mulloss/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ä
8atrain/gradients/loss/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs(atrain/gradients/loss/Minimum_grad/Shape*atrain/gradients/loss/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ć
)atrain/gradients/loss/Minimum_grad/SelectSelect,atrain/gradients/loss/Minimum_grad/LessEqual'atrain/gradients/loss/Mean_grad/truediv(atrain/gradients/loss/Minimum_grad/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
č
+atrain/gradients/loss/Minimum_grad/Select_1Select,atrain/gradients/loss/Minimum_grad/LessEqual(atrain/gradients/loss/Minimum_grad/zeros'atrain/gradients/loss/Mean_grad/truediv*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
&atrain/gradients/loss/Minimum_grad/SumSum)atrain/gradients/loss/Minimum_grad/Select8atrain/gradients/loss/Minimum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ç
*atrain/gradients/loss/Minimum_grad/ReshapeReshape&atrain/gradients/loss/Minimum_grad/Sum(atrain/gradients/loss/Minimum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
(atrain/gradients/loss/Minimum_grad/Sum_1Sum+atrain/gradients/loss/Minimum_grad/Select_1:atrain/gradients/loss/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Í
,atrain/gradients/loss/Minimum_grad/Reshape_1Reshape(atrain/gradients/loss/Minimum_grad/Sum_1*atrain/gradients/loss/Minimum_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

3atrain/gradients/loss/Minimum_grad/tuple/group_depsNoOp+^atrain/gradients/loss/Minimum_grad/Reshape-^atrain/gradients/loss/Minimum_grad/Reshape_1

;atrain/gradients/loss/Minimum_grad/tuple/control_dependencyIdentity*atrain/gradients/loss/Minimum_grad/Reshape4^atrain/gradients/loss/Minimum_grad/tuple/group_deps*
T0*=
_class3
1/loc:@atrain/gradients/loss/Minimum_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
=atrain/gradients/loss/Minimum_grad/tuple/control_dependency_1Identity,atrain/gradients/loss/Minimum_grad/Reshape_14^atrain/gradients/loss/Minimum_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@atrain/gradients/loss/Minimum_grad/Reshape_1

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
ö
>atrain/gradients/loss/surrogate/mul_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/loss/surrogate/mul_grad/Shape0atrain/gradients/loss/surrogate/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
­
,atrain/gradients/loss/surrogate/mul_grad/MulMul;atrain/gradients/loss/Minimum_grad/tuple/control_dependency	advantage*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
,atrain/gradients/loss/surrogate/mul_grad/SumSum,atrain/gradients/loss/surrogate/mul_grad/Mul>atrain/gradients/loss/surrogate/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ů
0atrain/gradients/loss/surrogate/mul_grad/ReshapeReshape,atrain/gradients/loss/surrogate/mul_grad/Sum.atrain/gradients/loss/surrogate/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
.atrain/gradients/loss/surrogate/mul_grad/Mul_1Mulloss/surrogate/truediv;atrain/gradients/loss/Minimum_grad/tuple/control_dependency*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ç
.atrain/gradients/loss/surrogate/mul_grad/Sum_1Sum.atrain/gradients/loss/surrogate/mul_grad/Mul_1@atrain/gradients/loss/surrogate/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ß
2atrain/gradients/loss/surrogate/mul_grad/Reshape_1Reshape.atrain/gradients/loss/surrogate/mul_grad/Sum_10atrain/gradients/loss/surrogate/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
9atrain/gradients/loss/surrogate/mul_grad/tuple/group_depsNoOp1^atrain/gradients/loss/surrogate/mul_grad/Reshape3^atrain/gradients/loss/surrogate/mul_grad/Reshape_1
˛
Aatrain/gradients/loss/surrogate/mul_grad/tuple/control_dependencyIdentity0atrain/gradients/loss/surrogate/mul_grad/Reshape:^atrain/gradients/loss/surrogate/mul_grad/tuple/group_deps*
T0*C
_class9
75loc:@atrain/gradients/loss/surrogate/mul_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
Catrain/gradients/loss/surrogate/mul_grad/tuple/control_dependency_1Identity2atrain/gradients/loss/surrogate/mul_grad/Reshape_1:^atrain/gradients/loss/surrogate/mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/loss/surrogate/mul_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
$atrain/gradients/loss/mul_grad/ShapeShapeloss/clip_by_value*
_output_shapes
:*
T0*
out_type0
o
&atrain/gradients/loss/mul_grad/Shape_1Shape	advantage*
T0*
out_type0*
_output_shapes
:
Ř
4atrain/gradients/loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs$atrain/gradients/loss/mul_grad/Shape&atrain/gradients/loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
"atrain/gradients/loss/mul_grad/MulMul=atrain/gradients/loss/Minimum_grad/tuple/control_dependency_1	advantage*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
"atrain/gradients/loss/mul_grad/SumSum"atrain/gradients/loss/mul_grad/Mul4atrain/gradients/loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ť
&atrain/gradients/loss/mul_grad/ReshapeReshape"atrain/gradients/loss/mul_grad/Sum$atrain/gradients/loss/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
$atrain/gradients/loss/mul_grad/Mul_1Mulloss/clip_by_value=atrain/gradients/loss/Minimum_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
$atrain/gradients/loss/mul_grad/Sum_1Sum$atrain/gradients/loss/mul_grad/Mul_16atrain/gradients/loss/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Á
(atrain/gradients/loss/mul_grad/Reshape_1Reshape$atrain/gradients/loss/mul_grad/Sum_1&atrain/gradients/loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

/atrain/gradients/loss/mul_grad/tuple/group_depsNoOp'^atrain/gradients/loss/mul_grad/Reshape)^atrain/gradients/loss/mul_grad/Reshape_1

7atrain/gradients/loss/mul_grad/tuple/control_dependencyIdentity&atrain/gradients/loss/mul_grad/Reshape0^atrain/gradients/loss/mul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*9
_class/
-+loc:@atrain/gradients/loss/mul_grad/Reshape

9atrain/gradients/loss/mul_grad/tuple/control_dependency_1Identity(atrain/gradients/loss/mul_grad/Reshape_10^atrain/gradients/loss/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@atrain/gradients/loss/mul_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

.atrain/gradients/loss/clip_by_value_grad/ShapeShapeloss/clip_by_value/Minimum*
T0*
out_type0*
_output_shapes
:
s
0atrain/gradients/loss/clip_by_value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
§
0atrain/gradients/loss/clip_by_value_grad/Shape_2Shape7atrain/gradients/loss/mul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
y
4atrain/gradients/loss/clip_by_value_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
â
.atrain/gradients/loss/clip_by_value_grad/zerosFill0atrain/gradients/loss/clip_by_value_grad/Shape_24atrain/gradients/loss/clip_by_value_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
5atrain/gradients/loss/clip_by_value_grad/GreaterEqualGreaterEqualloss/clip_by_value/Minimumloss/clip_by_value/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ö
>atrain/gradients/loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/loss/clip_by_value_grad/Shape0atrain/gradients/loss/clip_by_value_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

/atrain/gradients/loss/clip_by_value_grad/SelectSelect5atrain/gradients/loss/clip_by_value_grad/GreaterEqual7atrain/gradients/loss/mul_grad/tuple/control_dependency.atrain/gradients/loss/clip_by_value_grad/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

1atrain/gradients/loss/clip_by_value_grad/Select_1Select5atrain/gradients/loss/clip_by_value_grad/GreaterEqual.atrain/gradients/loss/clip_by_value_grad/zeros7atrain/gradients/loss/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
,atrain/gradients/loss/clip_by_value_grad/SumSum/atrain/gradients/loss/clip_by_value_grad/Select>atrain/gradients/loss/clip_by_value_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ů
0atrain/gradients/loss/clip_by_value_grad/ReshapeReshape,atrain/gradients/loss/clip_by_value_grad/Sum.atrain/gradients/loss/clip_by_value_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ę
.atrain/gradients/loss/clip_by_value_grad/Sum_1Sum1atrain/gradients/loss/clip_by_value_grad/Select_1@atrain/gradients/loss/clip_by_value_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Î
2atrain/gradients/loss/clip_by_value_grad/Reshape_1Reshape.atrain/gradients/loss/clip_by_value_grad/Sum_10atrain/gradients/loss/clip_by_value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Š
9atrain/gradients/loss/clip_by_value_grad/tuple/group_depsNoOp1^atrain/gradients/loss/clip_by_value_grad/Reshape3^atrain/gradients/loss/clip_by_value_grad/Reshape_1
˛
Aatrain/gradients/loss/clip_by_value_grad/tuple/control_dependencyIdentity0atrain/gradients/loss/clip_by_value_grad/Reshape:^atrain/gradients/loss/clip_by_value_grad/tuple/group_deps*
T0*C
_class9
75loc:@atrain/gradients/loss/clip_by_value_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
Catrain/gradients/loss/clip_by_value_grad/tuple/control_dependency_1Identity2atrain/gradients/loss/clip_by_value_grad/Reshape_1:^atrain/gradients/loss/clip_by_value_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/loss/clip_by_value_grad/Reshape_1*
_output_shapes
: 

6atrain/gradients/loss/clip_by_value/Minimum_grad/ShapeShapeloss/surrogate/truediv*
T0*
out_type0*
_output_shapes
:
{
8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
š
8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_2ShapeAatrain/gradients/loss/clip_by_value_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

<atrain/gradients/loss/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ú
6atrain/gradients/loss/clip_by_value/Minimum_grad/zerosFill8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_2<atrain/gradients/loss/clip_by_value/Minimum_grad/zeros/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
Ż
:atrain/gradients/loss/clip_by_value/Minimum_grad/LessEqual	LessEqualloss/surrogate/truedivloss/clip_by_value/Minimum/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Fatrain/gradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs6atrain/gradients/loss/clip_by_value/Minimum_grad/Shape8atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ş
7atrain/gradients/loss/clip_by_value/Minimum_grad/SelectSelect:atrain/gradients/loss/clip_by_value/Minimum_grad/LessEqualAatrain/gradients/loss/clip_by_value_grad/tuple/control_dependency6atrain/gradients/loss/clip_by_value/Minimum_grad/zeros*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
9atrain/gradients/loss/clip_by_value/Minimum_grad/Select_1Select:atrain/gradients/loss/clip_by_value/Minimum_grad/LessEqual6atrain/gradients/loss/clip_by_value/Minimum_grad/zerosAatrain/gradients/loss/clip_by_value_grad/tuple/control_dependency*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ü
4atrain/gradients/loss/clip_by_value/Minimum_grad/SumSum7atrain/gradients/loss/clip_by_value/Minimum_grad/SelectFatrain/gradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ń
8atrain/gradients/loss/clip_by_value/Minimum_grad/ReshapeReshape4atrain/gradients/loss/clip_by_value/Minimum_grad/Sum6atrain/gradients/loss/clip_by_value/Minimum_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

6atrain/gradients/loss/clip_by_value/Minimum_grad/Sum_1Sum9atrain/gradients/loss/clip_by_value/Minimum_grad/Select_1Hatrain/gradients/loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ć
:atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1Reshape6atrain/gradients/loss/clip_by_value/Minimum_grad/Sum_18atrain/gradients/loss/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Á
Aatrain/gradients/loss/clip_by_value/Minimum_grad/tuple/group_depsNoOp9^atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape;^atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1
Ň
Iatrain/gradients/loss/clip_by_value/Minimum_grad/tuple/control_dependencyIdentity8atrain/gradients/loss/clip_by_value/Minimum_grad/ReshapeB^atrain/gradients/loss/clip_by_value/Minimum_grad/tuple/group_deps*
T0*K
_classA
?=loc:@atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
Katrain/gradients/loss/clip_by_value/Minimum_grad/tuple/control_dependency_1Identity:atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1B^atrain/gradients/loss/clip_by_value/Minimum_grad/tuple/group_deps*
T0*M
_classC
A?loc:@atrain/gradients/loss/clip_by_value/Minimum_grad/Reshape_1*
_output_shapes
: 
Ť
atrain/gradients/AddNAddNAatrain/gradients/loss/surrogate/mul_grad/tuple/control_dependencyIatrain/gradients/loss/clip_by_value/Minimum_grad/tuple/control_dependency*
T0*C
_class9
75loc:@atrain/gradients/loss/surrogate/mul_grad/Reshape*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

2atrain/gradients/loss/surrogate/truediv_grad/ShapeShapepi/Normal/prob/Exp*
T0*
out_type0*
_output_shapes
:

4atrain/gradients/loss/surrogate/truediv_grad/Shape_1Shapeoldpi/Normal/prob/Exp*
T0*
out_type0*
_output_shapes
:

Batrain/gradients/loss/surrogate/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs2atrain/gradients/loss/surrogate/truediv_grad/Shape4atrain/gradients/loss/surrogate/truediv_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

4atrain/gradients/loss/surrogate/truediv_grad/RealDivRealDivatrain/gradients/AddNoldpi/Normal/prob/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ń
0atrain/gradients/loss/surrogate/truediv_grad/SumSum4atrain/gradients/loss/surrogate/truediv_grad/RealDivBatrain/gradients/loss/surrogate/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ĺ
4atrain/gradients/loss/surrogate/truediv_grad/ReshapeReshape0atrain/gradients/loss/surrogate/truediv_grad/Sum2atrain/gradients/loss/surrogate/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
0atrain/gradients/loss/surrogate/truediv_grad/NegNegpi/Normal/prob/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ź
6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_1RealDiv0atrain/gradients/loss/surrogate/truediv_grad/Negoldpi/Normal/prob/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_2RealDiv6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_1oldpi/Normal/prob/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
0atrain/gradients/loss/surrogate/truediv_grad/mulMulatrain/gradients/AddN6atrain/gradients/loss/surrogate/truediv_grad/RealDiv_2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ń
2atrain/gradients/loss/surrogate/truediv_grad/Sum_1Sum0atrain/gradients/loss/surrogate/truediv_grad/mulDatrain/gradients/loss/surrogate/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ë
6atrain/gradients/loss/surrogate/truediv_grad/Reshape_1Reshape2atrain/gradients/loss/surrogate/truediv_grad/Sum_14atrain/gradients/loss/surrogate/truediv_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ľ
=atrain/gradients/loss/surrogate/truediv_grad/tuple/group_depsNoOp5^atrain/gradients/loss/surrogate/truediv_grad/Reshape7^atrain/gradients/loss/surrogate/truediv_grad/Reshape_1
Â
Eatrain/gradients/loss/surrogate/truediv_grad/tuple/control_dependencyIdentity4atrain/gradients/loss/surrogate/truediv_grad/Reshape>^atrain/gradients/loss/surrogate/truediv_grad/tuple/group_deps*
T0*G
_class=
;9loc:@atrain/gradients/loss/surrogate/truediv_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
Gatrain/gradients/loss/surrogate/truediv_grad/tuple/control_dependency_1Identity6atrain/gradients/loss/surrogate/truediv_grad/Reshape_1>^atrain/gradients/loss/surrogate/truediv_grad/tuple/group_deps*
T0*I
_class?
=;loc:@atrain/gradients/loss/surrogate/truediv_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
,atrain/gradients/pi/Normal/prob/Exp_grad/mulMulEatrain/gradients/loss/surrogate/truediv_grad/tuple/control_dependencypi/Normal/prob/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

.atrain/gradients/pi/Normal/prob/sub_grad/ShapeShapepi/Normal/prob/mul*
out_type0*
_output_shapes
:*
T0

0atrain/gradients/pi/Normal/prob/sub_grad/Shape_1Shapepi/Normal/prob/add*
T0*
out_type0*
_output_shapes
:
ö
>atrain/gradients/pi/Normal/prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/pi/Normal/prob/sub_grad/Shape0atrain/gradients/pi/Normal/prob/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
á
,atrain/gradients/pi/Normal/prob/sub_grad/SumSum,atrain/gradients/pi/Normal/prob/Exp_grad/mul>atrain/gradients/pi/Normal/prob/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ů
0atrain/gradients/pi/Normal/prob/sub_grad/ReshapeReshape,atrain/gradients/pi/Normal/prob/sub_grad/Sum.atrain/gradients/pi/Normal/prob/sub_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ĺ
.atrain/gradients/pi/Normal/prob/sub_grad/Sum_1Sum,atrain/gradients/pi/Normal/prob/Exp_grad/mul@atrain/gradients/pi/Normal/prob/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

,atrain/gradients/pi/Normal/prob/sub_grad/NegNeg.atrain/gradients/pi/Normal/prob/sub_grad/Sum_1*
T0*
_output_shapes
:
Ý
2atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1Reshape,atrain/gradients/pi/Normal/prob/sub_grad/Neg0atrain/gradients/pi/Normal/prob/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
9atrain/gradients/pi/Normal/prob/sub_grad/tuple/group_depsNoOp1^atrain/gradients/pi/Normal/prob/sub_grad/Reshape3^atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1
˛
Aatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependencyIdentity0atrain/gradients/pi/Normal/prob/sub_grad/Reshape:^atrain/gradients/pi/Normal/prob/sub_grad/tuple/group_deps*
T0*C
_class9
75loc:@atrain/gradients/pi/Normal/prob/sub_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
Catrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1:^atrain/gradients/pi/Normal/prob/sub_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/pi/Normal/prob/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
.atrain/gradients/pi/Normal/prob/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

0atrain/gradients/pi/Normal/prob/mul_grad/Shape_1Shapepi/Normal/prob/Square*
T0*
out_type0*
_output_shapes
:
ö
>atrain/gradients/pi/Normal/prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/pi/Normal/prob/mul_grad/Shape0atrain/gradients/pi/Normal/prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ż
,atrain/gradients/pi/Normal/prob/mul_grad/MulMulAatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependencypi/Normal/prob/Square*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
á
,atrain/gradients/pi/Normal/prob/mul_grad/SumSum,atrain/gradients/pi/Normal/prob/mul_grad/Mul>atrain/gradients/pi/Normal/prob/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Č
0atrain/gradients/pi/Normal/prob/mul_grad/ReshapeReshape,atrain/gradients/pi/Normal/prob/mul_grad/Sum.atrain/gradients/pi/Normal/prob/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ŕ
.atrain/gradients/pi/Normal/prob/mul_grad/Mul_1Mulpi/Normal/prob/mul/xAatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ç
.atrain/gradients/pi/Normal/prob/mul_grad/Sum_1Sum.atrain/gradients/pi/Normal/prob/mul_grad/Mul_1@atrain/gradients/pi/Normal/prob/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ß
2atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1Reshape.atrain/gradients/pi/Normal/prob/mul_grad/Sum_10atrain/gradients/pi/Normal/prob/mul_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Š
9atrain/gradients/pi/Normal/prob/mul_grad/tuple/group_depsNoOp1^atrain/gradients/pi/Normal/prob/mul_grad/Reshape3^atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1
Ą
Aatrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependencyIdentity0atrain/gradients/pi/Normal/prob/mul_grad/Reshape:^atrain/gradients/pi/Normal/prob/mul_grad/tuple/group_deps*C
_class9
75loc:@atrain/gradients/pi/Normal/prob/mul_grad/Reshape*
_output_shapes
: *
T0
¸
Catrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1:^atrain/gradients/pi/Normal/prob/mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/pi/Normal/prob/mul_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
.atrain/gradients/pi/Normal/prob/add_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

0atrain/gradients/pi/Normal/prob/add_grad/Shape_1Shapepi/Normal/prob/Log*
_output_shapes
:*
T0*
out_type0
ö
>atrain/gradients/pi/Normal/prob/add_grad/BroadcastGradientArgsBroadcastGradientArgs.atrain/gradients/pi/Normal/prob/add_grad/Shape0atrain/gradients/pi/Normal/prob/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ř
,atrain/gradients/pi/Normal/prob/add_grad/SumSumCatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency_1>atrain/gradients/pi/Normal/prob/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Č
0atrain/gradients/pi/Normal/prob/add_grad/ReshapeReshape,atrain/gradients/pi/Normal/prob/add_grad/Sum.atrain/gradients/pi/Normal/prob/add_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ü
.atrain/gradients/pi/Normal/prob/add_grad/Sum_1SumCatrain/gradients/pi/Normal/prob/sub_grad/tuple/control_dependency_1@atrain/gradients/pi/Normal/prob/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ß
2atrain/gradients/pi/Normal/prob/add_grad/Reshape_1Reshape.atrain/gradients/pi/Normal/prob/add_grad/Sum_10atrain/gradients/pi/Normal/prob/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
9atrain/gradients/pi/Normal/prob/add_grad/tuple/group_depsNoOp1^atrain/gradients/pi/Normal/prob/add_grad/Reshape3^atrain/gradients/pi/Normal/prob/add_grad/Reshape_1
Ą
Aatrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependencyIdentity0atrain/gradients/pi/Normal/prob/add_grad/Reshape:^atrain/gradients/pi/Normal/prob/add_grad/tuple/group_deps*
T0*C
_class9
75loc:@atrain/gradients/pi/Normal/prob/add_grad/Reshape*
_output_shapes
: 
¸
Catrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/Normal/prob/add_grad/Reshape_1:^atrain/gradients/pi/Normal/prob/add_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*E
_class;
97loc:@atrain/gradients/pi/Normal/prob/add_grad/Reshape_1
ź
1atrain/gradients/pi/Normal/prob/Square_grad/ConstConstD^atrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
ż
/atrain/gradients/pi/Normal/prob/Square_grad/MulMul"pi/Normal/prob/standardize/truediv1atrain/gradients/pi/Normal/prob/Square_grad/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ŕ
1atrain/gradients/pi/Normal/prob/Square_grad/Mul_1MulCatrain/gradients/pi/Normal/prob/mul_grad/tuple/control_dependency_1/atrain/gradients/pi/Normal/prob/Square_grad/Mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ę
3atrain/gradients/pi/Normal/prob/Log_grad/Reciprocal
Reciprocalpi/Normal/scaleD^atrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
,atrain/gradients/pi/Normal/prob/Log_grad/mulMulCatrain/gradients/pi/Normal/prob/add_grad/tuple/control_dependency_13atrain/gradients/pi/Normal/prob/Log_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ShapeShapepi/Normal/prob/standardize/sub*
T0*
out_type0*
_output_shapes
:

@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape_1Shapepi/Normal/scale*
T0*
out_type0*
_output_shapes
:
Ś
Natrain/gradients/pi/Normal/prob/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Á
@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDivRealDiv1atrain/gradients/pi/Normal/prob/Square_grad/Mul_1pi/Normal/scale*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/SumSum@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDivNatrain/gradients/pi/Normal/prob/standardize/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ReshapeReshape<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Sum>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/NegNegpi/Normal/prob/standardize/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Î
Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_1RealDiv<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Negpi/Normal/scale*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ô
Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_2RealDivBatrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_1pi/Normal/scale*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ě
<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/mulMul1atrain/gradients/pi/Normal/prob/Square_grad/Mul_1Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Sum_1Sum<atrain/gradients/pi/Normal/prob/standardize/truediv_grad/mulPatrain/gradients/pi/Normal/prob/standardize/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Batrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1Reshape>atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Sum_1@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ů
Iatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/group_depsNoOpA^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ReshapeC^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1
ň
Qatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependencyIdentity@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/ReshapeJ^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/group_deps*
T0*S
_classI
GEloc:@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ř
Satrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependency_1IdentityBatrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1J^atrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*U
_classK
IGloc:@atrain/gradients/pi/Normal/prob/standardize/truediv_grad/Reshape_1

:atrain/gradients/pi/Normal/prob/standardize/sub_grad/ShapeShapeaction*
out_type0*
_output_shapes
:*
T0

<atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape_1Shapepi/Normal/loc*
T0*
out_type0*
_output_shapes
:

Jatrain/gradients/pi/Normal/prob/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape<atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

8atrain/gradients/pi/Normal/prob/standardize/sub_grad/SumSumQatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependencyJatrain/gradients/pi/Normal/prob/standardize/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ý
<atrain/gradients/pi/Normal/prob/standardize/sub_grad/ReshapeReshape8atrain/gradients/pi/Normal/prob/standardize/sub_grad/Sum:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Sum_1SumQatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependencyLatrain/gradients/pi/Normal/prob/standardize/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

8atrain/gradients/pi/Normal/prob/standardize/sub_grad/NegNeg:atrain/gradients/pi/Normal/prob/standardize/sub_grad/Sum_1*
T0*
_output_shapes
:

>atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1Reshape8atrain/gradients/pi/Normal/prob/standardize/sub_grad/Neg<atrain/gradients/pi/Normal/prob/standardize/sub_grad/Shape_1*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Í
Eatrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/group_depsNoOp=^atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape?^atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1
â
Matrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/control_dependencyIdentity<atrain/gradients/pi/Normal/prob/standardize/sub_grad/ReshapeF^atrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/group_deps*
T0*O
_classE
CAloc:@atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
Oatrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/control_dependency_1Identity>atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1F^atrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@atrain/gradients/pi/Normal/prob/standardize/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

atrain/gradients/AddN_1AddN,atrain/gradients/pi/Normal/prob/Log_grad/mulSatrain/gradients/pi/Normal/prob/standardize/truediv_grad/tuple/control_dependency_1*
T0*?
_class5
31loc:@atrain/gradients/pi/Normal/prob/Log_grad/mul*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
+atrain/gradients/pi/scaled_sigma_grad/ShapeShapepi/dense/Softplus*
_output_shapes
:*
T0*
out_type0
w
-atrain/gradients/pi/scaled_sigma_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
í
;atrain/gradients/pi/scaled_sigma_grad/BroadcastGradientArgsBroadcastGradientArgs+atrain/gradients/pi/scaled_sigma_grad/Shape-atrain/gradients/pi/scaled_sigma_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

)atrain/gradients/pi/scaled_sigma_grad/MulMulatrain/gradients/AddN_1pi/scaled_sigma/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
)atrain/gradients/pi/scaled_sigma_grad/SumSum)atrain/gradients/pi/scaled_sigma_grad/Mul;atrain/gradients/pi/scaled_sigma_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Đ
-atrain/gradients/pi/scaled_sigma_grad/ReshapeReshape)atrain/gradients/pi/scaled_sigma_grad/Sum+atrain/gradients/pi/scaled_sigma_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

+atrain/gradients/pi/scaled_sigma_grad/Mul_1Mulpi/dense/Softplusatrain/gradients/AddN_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ţ
+atrain/gradients/pi/scaled_sigma_grad/Sum_1Sum+atrain/gradients/pi/scaled_sigma_grad/Mul_1=atrain/gradients/pi/scaled_sigma_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
É
/atrain/gradients/pi/scaled_sigma_grad/Reshape_1Reshape+atrain/gradients/pi/scaled_sigma_grad/Sum_1-atrain/gradients/pi/scaled_sigma_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
 
6atrain/gradients/pi/scaled_sigma_grad/tuple/group_depsNoOp.^atrain/gradients/pi/scaled_sigma_grad/Reshape0^atrain/gradients/pi/scaled_sigma_grad/Reshape_1
Ś
>atrain/gradients/pi/scaled_sigma_grad/tuple/control_dependencyIdentity-atrain/gradients/pi/scaled_sigma_grad/Reshape7^atrain/gradients/pi/scaled_sigma_grad/tuple/group_deps*
T0*@
_class6
42loc:@atrain/gradients/pi/scaled_sigma_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

@atrain/gradients/pi/scaled_sigma_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/scaled_sigma_grad/Reshape_17^atrain/gradients/pi/scaled_sigma_grad/tuple/group_deps*
T0*B
_class8
64loc:@atrain/gradients/pi/scaled_sigma_grad/Reshape_1*
_output_shapes
:
q
(atrain/gradients/pi/scaled_mu_grad/ShapeShape	pi/a/Tanh*
_output_shapes
:*
T0*
out_type0
t
*atrain/gradients/pi/scaled_mu_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
ä
8atrain/gradients/pi/scaled_mu_grad/BroadcastGradientArgsBroadcastGradientArgs(atrain/gradients/pi/scaled_mu_grad/Shape*atrain/gradients/pi/scaled_mu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ŕ
&atrain/gradients/pi/scaled_mu_grad/MulMulOatrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/control_dependency_1pi/scaled_mu/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
&atrain/gradients/pi/scaled_mu_grad/SumSum&atrain/gradients/pi/scaled_mu_grad/Mul8atrain/gradients/pi/scaled_mu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ç
*atrain/gradients/pi/scaled_mu_grad/ReshapeReshape&atrain/gradients/pi/scaled_mu_grad/Sum(atrain/gradients/pi/scaled_mu_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
(atrain/gradients/pi/scaled_mu_grad/Mul_1Mul	pi/a/TanhOatrain/gradients/pi/Normal/prob/standardize/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
(atrain/gradients/pi/scaled_mu_grad/Sum_1Sum(atrain/gradients/pi/scaled_mu_grad/Mul_1:atrain/gradients/pi/scaled_mu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ŕ
,atrain/gradients/pi/scaled_mu_grad/Reshape_1Reshape(atrain/gradients/pi/scaled_mu_grad/Sum_1*atrain/gradients/pi/scaled_mu_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

3atrain/gradients/pi/scaled_mu_grad/tuple/group_depsNoOp+^atrain/gradients/pi/scaled_mu_grad/Reshape-^atrain/gradients/pi/scaled_mu_grad/Reshape_1

;atrain/gradients/pi/scaled_mu_grad/tuple/control_dependencyIdentity*atrain/gradients/pi/scaled_mu_grad/Reshape4^atrain/gradients/pi/scaled_mu_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*=
_class3
1/loc:@atrain/gradients/pi/scaled_mu_grad/Reshape

=atrain/gradients/pi/scaled_mu_grad/tuple/control_dependency_1Identity,atrain/gradients/pi/scaled_mu_grad/Reshape_14^atrain/gradients/pi/scaled_mu_grad/tuple/group_deps*
_output_shapes
:*
T0*?
_class5
31loc:@atrain/gradients/pi/scaled_mu_grad/Reshape_1
Č
4atrain/gradients/pi/dense/Softplus_grad/SoftplusGradSoftplusGrad>atrain/gradients/pi/scaled_sigma_grad/tuple/control_dependencypi/dense/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
(atrain/gradients/pi/a/Tanh_grad/TanhGradTanhGrad	pi/a/Tanh;atrain/gradients/pi/scaled_mu_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
2atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad4atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad*
T0*
data_formatNHWC*
_output_shapes
:
Ť
7atrain/gradients/pi/dense/BiasAdd_grad/tuple/group_depsNoOp3^atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad
ś
?atrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity4atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad8^atrain/gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@atrain/gradients/pi/dense/Softplus_grad/SoftplusGrad*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
Aatrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity2atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGrad8^atrain/gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@atrain/gradients/pi/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ł
.atrain/gradients/pi/a/BiasAdd_grad/BiasAddGradBiasAddGrad(atrain/gradients/pi/a/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes
:*
T0

3atrain/gradients/pi/a/BiasAdd_grad/tuple/group_depsNoOp/^atrain/gradients/pi/a/BiasAdd_grad/BiasAddGrad)^atrain/gradients/pi/a/Tanh_grad/TanhGrad

;atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependencyIdentity(atrain/gradients/pi/a/Tanh_grad/TanhGrad4^atrain/gradients/pi/a/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*;
_class1
/-loc:@atrain/gradients/pi/a/Tanh_grad/TanhGrad

=atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependency_1Identity.atrain/gradients/pi/a/BiasAdd_grad/BiasAddGrad4^atrain/gradients/pi/a/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@atrain/gradients/pi/a/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
ć
,atrain/gradients/pi/dense/MatMul_grad/MatMulMatMul?atrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependencypi/dense/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ő
.atrain/gradients/pi/dense/MatMul_grad/MatMul_1MatMul
pi/l4/Relu?atrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0

6atrain/gradients/pi/dense/MatMul_grad/tuple/group_depsNoOp-^atrain/gradients/pi/dense/MatMul_grad/MatMul/^atrain/gradients/pi/dense/MatMul_grad/MatMul_1
Ľ
>atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependencyIdentity,atrain/gradients/pi/dense/MatMul_grad/MatMul7^atrain/gradients/pi/dense/MatMul_grad/tuple/group_deps*?
_class5
31loc:@atrain/gradients/pi/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˘
@atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Identity.atrain/gradients/pi/dense/MatMul_grad/MatMul_17^atrain/gradients/pi/dense/MatMul_grad/tuple/group_deps*A
_class7
53loc:@atrain/gradients/pi/dense/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
Ú
(atrain/gradients/pi/a/MatMul_grad/MatMulMatMul;atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependencypi/a/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Í
*atrain/gradients/pi/a/MatMul_grad/MatMul_1MatMul
pi/l4/Relu;atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 

2atrain/gradients/pi/a/MatMul_grad/tuple/group_depsNoOp)^atrain/gradients/pi/a/MatMul_grad/MatMul+^atrain/gradients/pi/a/MatMul_grad/MatMul_1

:atrain/gradients/pi/a/MatMul_grad/tuple/control_dependencyIdentity(atrain/gradients/pi/a/MatMul_grad/MatMul3^atrain/gradients/pi/a/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*;
_class1
/-loc:@atrain/gradients/pi/a/MatMul_grad/MatMul

<atrain/gradients/pi/a/MatMul_grad/tuple/control_dependency_1Identity*atrain/gradients/pi/a/MatMul_grad/MatMul_13^atrain/gradients/pi/a/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@atrain/gradients/pi/a/MatMul_grad/MatMul_1*
_output_shapes
:	

atrain/gradients/AddN_2AddN>atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependency:atrain/gradients/pi/a/MatMul_grad/tuple/control_dependency*?
_class5
31loc:@atrain/gradients/pi/dense/MatMul_grad/MatMul*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

)atrain/gradients/pi/l4/Relu_grad/ReluGradReluGradatrain/gradients/AddN_2
pi/l4/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
/atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

4atrain/gradients/pi/l4/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l4/Relu_grad/ReluGrad

<atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l4/Relu_grad/ReluGrad5^atrain/gradients/pi/l4/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*<
_class2
0.loc:@atrain/gradients/pi/l4/Relu_grad/ReluGrad

>atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l4/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*B
_class8
64loc:@atrain/gradients/pi/l4/BiasAdd_grad/BiasAddGrad
Ý
)atrain/gradients/pi/l4/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependencypi/l4/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Đ
+atrain/gradients/pi/l4/MatMul_grad/MatMul_1MatMul
pi/l3/Relu<atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

3atrain/gradients/pi/l4/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l4/MatMul_grad/MatMul,^atrain/gradients/pi/l4/MatMul_grad/MatMul_1

;atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l4/MatMul_grad/MatMul4^atrain/gradients/pi/l4/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l4/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l4/MatMul_grad/MatMul_14^atrain/gradients/pi/l4/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@atrain/gradients/pi/l4/MatMul_grad/MatMul_1* 
_output_shapes
:

ą
)atrain/gradients/pi/l3/Relu_grad/ReluGradReluGrad;atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependency
pi/l3/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
/atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l3/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*
data_formatNHWC

4atrain/gradients/pi/l3/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l3/Relu_grad/ReluGrad

<atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l3/Relu_grad/ReluGrad5^atrain/gradients/pi/l3/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*<
_class2
0.loc:@atrain/gradients/pi/l3/Relu_grad/ReluGrad

>atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l3/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@atrain/gradients/pi/l3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ý
)atrain/gradients/pi/l3/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependencypi/l3/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Đ
+atrain/gradients/pi/l3/MatMul_grad/MatMul_1MatMul
pi/l2/Relu<atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

3atrain/gradients/pi/l3/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l3/MatMul_grad/MatMul,^atrain/gradients/pi/l3/MatMul_grad/MatMul_1

;atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l3/MatMul_grad/MatMul4^atrain/gradients/pi/l3/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@atrain/gradients/pi/l3/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l3/MatMul_grad/MatMul_14^atrain/gradients/pi/l3/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*>
_class4
20loc:@atrain/gradients/pi/l3/MatMul_grad/MatMul_1
ą
)atrain/gradients/pi/l2/Relu_grad/ReluGradReluGrad;atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependency
pi/l2/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
/atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0

4atrain/gradients/pi/l2/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l2/Relu_grad/ReluGrad

<atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l2/Relu_grad/ReluGrad5^atrain/gradients/pi/l2/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@atrain/gradients/pi/l2/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

>atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l2/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@atrain/gradients/pi/l2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ý
)atrain/gradients/pi/l2/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependencypi/l2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Đ
+atrain/gradients/pi/l2/MatMul_grad/MatMul_1MatMul
pi/l1/Relu<atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(

3atrain/gradients/pi/l2/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l2/MatMul_grad/MatMul,^atrain/gradients/pi/l2/MatMul_grad/MatMul_1

;atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l2/MatMul_grad/MatMul4^atrain/gradients/pi/l2/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*<
_class2
0.loc:@atrain/gradients/pi/l2/MatMul_grad/MatMul

=atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l2/MatMul_grad/MatMul_14^atrain/gradients/pi/l2/MatMul_grad/tuple/group_deps*>
_class4
20loc:@atrain/gradients/pi/l2/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
ą
)atrain/gradients/pi/l1/Relu_grad/ReluGradReluGrad;atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependency
pi/l1/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
/atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGradBiasAddGrad)atrain/gradients/pi/l1/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*
data_formatNHWC

4atrain/gradients/pi/l1/BiasAdd_grad/tuple/group_depsNoOp0^atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGrad*^atrain/gradients/pi/l1/Relu_grad/ReluGrad

<atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l1/Relu_grad/ReluGrad5^atrain/gradients/pi/l1/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@atrain/gradients/pi/l1/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

>atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependency_1Identity/atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGrad5^atrain/gradients/pi/l1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*B
_class8
64loc:@atrain/gradients/pi/l1/BiasAdd_grad/BiasAddGrad
Ü
)atrain/gradients/pi/l1/MatMul_grad/MatMulMatMul<atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependencypi/l1/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Ę
+atrain/gradients/pi/l1/MatMul_grad/MatMul_1MatMulstate<atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 

3atrain/gradients/pi/l1/MatMul_grad/tuple/group_depsNoOp*^atrain/gradients/pi/l1/MatMul_grad/MatMul,^atrain/gradients/pi/l1/MatMul_grad/MatMul_1

;atrain/gradients/pi/l1/MatMul_grad/tuple/control_dependencyIdentity)atrain/gradients/pi/l1/MatMul_grad/MatMul4^atrain/gradients/pi/l1/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@atrain/gradients/pi/l1/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

=atrain/gradients/pi/l1/MatMul_grad/tuple/control_dependency_1Identity+atrain/gradients/pi/l1/MatMul_grad/MatMul_14^atrain/gradients/pi/l1/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@atrain/gradients/pi/l1/MatMul_grad/MatMul_1*
_output_shapes
:	

 atrain/beta1_power/initial_valueConst*
_class
loc:@pi/a/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 

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
Á
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

 atrain/beta2_power/initial_valueConst*
_class
loc:@pi/a/bias*
valueB
 *wž?*
dtype0*
_output_shapes
: 

atrain/beta2_power
VariableV2*
_output_shapes
: *
shared_name *
_class
loc:@pi/a/bias*
	container *
shape: *
dtype0
Á
atrain/beta2_power/AssignAssignatrain/beta2_power atrain/beta2_power/initial_value*
use_locking(*
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
: 
v
atrain/beta2_power/readIdentityatrain/beta2_power*
_class
loc:@pi/a/bias*
_output_shapes
: *
T0
Ź
:atrain/pi/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@pi/l1/kernel*
valueB"      

0atrain/pi/l1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
_class
loc:@pi/l1/kernel*
valueB
 *    *
dtype0
ý
*atrain/pi/l1/kernel/Adam/Initializer/zerosFill:atrain/pi/l1/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l1/kernel/Adam/Initializer/zeros/Const*
T0*
_class
loc:@pi/l1/kernel*

index_type0*
_output_shapes
:	
Ż
atrain/pi/l1/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *
_class
loc:@pi/l1/kernel*
	container *
shape:	
ă
atrain/pi/l1/kernel/Adam/AssignAssignatrain/pi/l1/kernel/Adam*atrain/pi/l1/kernel/Adam/Initializer/zeros*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@pi/l1/kernel*
validate_shape(

atrain/pi/l1/kernel/Adam/readIdentityatrain/pi/l1/kernel/Adam*
_output_shapes
:	*
T0*
_class
loc:@pi/l1/kernel
Ž
<atrain/pi/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l1/kernel*
valueB"      *
dtype0*
_output_shapes
:

2atrain/pi/l1/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@pi/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,atrain/pi/l1/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l1/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	*
T0*
_class
loc:@pi/l1/kernel*

index_type0
ą
atrain/pi/l1/kernel/Adam_1
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *
_class
loc:@pi/l1/kernel*
	container 
é
!atrain/pi/l1/kernel/Adam_1/AssignAssignatrain/pi/l1/kernel/Adam_1,atrain/pi/l1/kernel/Adam_1/Initializer/zeros*
_class
loc:@pi/l1/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0

atrain/pi/l1/kernel/Adam_1/readIdentityatrain/pi/l1/kernel/Adam_1*
T0*
_class
loc:@pi/l1/kernel*
_output_shapes
:	

(atrain/pi/l1/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
_class
loc:@pi/l1/bias*
valueB*    
Ł
atrain/pi/l1/bias/Adam
VariableV2*
shared_name *
_class
loc:@pi/l1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
×
atrain/pi/l1/bias/Adam/AssignAssignatrain/pi/l1/bias/Adam(atrain/pi/l1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@pi/l1/bias

atrain/pi/l1/bias/Adam/readIdentityatrain/pi/l1/bias/Adam*
T0*
_class
loc:@pi/l1/bias*
_output_shapes	
:

*atrain/pi/l1/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ľ
atrain/pi/l1/bias/Adam_1
VariableV2*
_class
loc:@pi/l1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ý
atrain/pi/l1/bias/Adam_1/AssignAssignatrain/pi/l1/bias/Adam_1*atrain/pi/l1/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@pi/l1/bias*
validate_shape(

atrain/pi/l1/bias/Adam_1/readIdentityatrain/pi/l1/bias/Adam_1*
_output_shapes	
:*
T0*
_class
loc:@pi/l1/bias
Ź
:atrain/pi/l2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:

0atrain/pi/l2/kernel/Adam/Initializer/zeros/ConstConst*
_class
loc:@pi/l2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ţ
*atrain/pi/l2/kernel/Adam/Initializer/zerosFill:atrain/pi/l2/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l2/kernel/Adam/Initializer/zeros/Const*
T0*
_class
loc:@pi/l2/kernel*

index_type0* 
_output_shapes
:

ą
atrain/pi/l2/kernel/Adam
VariableV2*
shared_name *
_class
loc:@pi/l2/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ä
atrain/pi/l2/kernel/Adam/AssignAssignatrain/pi/l2/kernel/Adam*atrain/pi/l2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l2/kernel*
validate_shape(* 
_output_shapes
:


atrain/pi/l2/kernel/Adam/readIdentityatrain/pi/l2/kernel/Adam*
_class
loc:@pi/l2/kernel* 
_output_shapes
:
*
T0
Ž
<atrain/pi/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l2/kernel*
valueB"      *
dtype0*
_output_shapes
:

2atrain/pi/l2/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@pi/l2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,atrain/pi/l2/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l2/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l2/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*
_class
loc:@pi/l2/kernel*

index_type0
ł
atrain/pi/l2/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *
_class
loc:@pi/l2/kernel*
	container *
shape:

ę
!atrain/pi/l2/kernel/Adam_1/AssignAssignatrain/pi/l2/kernel/Adam_1,atrain/pi/l2/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@pi/l2/kernel*
validate_shape(

atrain/pi/l2/kernel/Adam_1/readIdentityatrain/pi/l2/kernel/Adam_1*
_class
loc:@pi/l2/kernel* 
_output_shapes
:
*
T0

(atrain/pi/l2/bias/Adam/Initializer/zerosConst*
_class
loc:@pi/l2/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ł
atrain/pi/l2/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l2/bias*
	container *
shape:
×
atrain/pi/l2/bias/Adam/AssignAssignatrain/pi/l2/bias/Adam(atrain/pi/l2/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l2/bias*
validate_shape(*
_output_shapes	
:

atrain/pi/l2/bias/Adam/readIdentityatrain/pi/l2/bias/Adam*
T0*
_class
loc:@pi/l2/bias*
_output_shapes	
:

*atrain/pi/l2/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l2/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ľ
atrain/pi/l2/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l2/bias
Ý
atrain/pi/l2/bias/Adam_1/AssignAssignatrain/pi/l2/bias/Adam_1*atrain/pi/l2/bias/Adam_1/Initializer/zeros*
_class
loc:@pi/l2/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

atrain/pi/l2/bias/Adam_1/readIdentityatrain/pi/l2/bias/Adam_1*
T0*
_class
loc:@pi/l2/bias*
_output_shapes	
:
Ź
:atrain/pi/l3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@pi/l3/kernel*
valueB"      

0atrain/pi/l3/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
_class
loc:@pi/l3/kernel*
valueB
 *    *
dtype0
ţ
*atrain/pi/l3/kernel/Adam/Initializer/zerosFill:atrain/pi/l3/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l3/kernel/Adam/Initializer/zeros/Const*
T0*
_class
loc:@pi/l3/kernel*

index_type0* 
_output_shapes
:

ą
atrain/pi/l3/kernel/Adam
VariableV2*
_class
loc:@pi/l3/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
ä
atrain/pi/l3/kernel/Adam/AssignAssignatrain/pi/l3/kernel/Adam*atrain/pi/l3/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l3/kernel*
validate_shape(* 
_output_shapes
:


atrain/pi/l3/kernel/Adam/readIdentityatrain/pi/l3/kernel/Adam*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:

Ž
<atrain/pi/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@pi/l3/kernel*
valueB"      

2atrain/pi/l3/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@pi/l3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,atrain/pi/l3/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l3/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l3/kernel/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@pi/l3/kernel*

index_type0* 
_output_shapes
:

ł
atrain/pi/l3/kernel/Adam_1
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *
_class
loc:@pi/l3/kernel
ę
!atrain/pi/l3/kernel/Adam_1/AssignAssignatrain/pi/l3/kernel/Adam_1,atrain/pi/l3/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@pi/l3/kernel*
validate_shape(

atrain/pi/l3/kernel/Adam_1/readIdentityatrain/pi/l3/kernel/Adam_1*
T0*
_class
loc:@pi/l3/kernel* 
_output_shapes
:


(atrain/pi/l3/bias/Adam/Initializer/zerosConst*
_class
loc:@pi/l3/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ł
atrain/pi/l3/bias/Adam
VariableV2*
_class
loc:@pi/l3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
×
atrain/pi/l3/bias/Adam/AssignAssignatrain/pi/l3/bias/Adam(atrain/pi/l3/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l3/bias*
validate_shape(*
_output_shapes	
:

atrain/pi/l3/bias/Adam/readIdentityatrain/pi/l3/bias/Adam*
_class
loc:@pi/l3/bias*
_output_shapes	
:*
T0

*atrain/pi/l3/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l3/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ľ
atrain/pi/l3/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l3/bias*
	container 
Ý
atrain/pi/l3/bias/Adam_1/AssignAssignatrain/pi/l3/bias/Adam_1*atrain/pi/l3/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@pi/l3/bias*
validate_shape(

atrain/pi/l3/bias/Adam_1/readIdentityatrain/pi/l3/bias/Adam_1*
T0*
_class
loc:@pi/l3/bias*
_output_shapes	
:
Ź
:atrain/pi/l4/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@pi/l4/kernel*
valueB"      *
dtype0*
_output_shapes
:

0atrain/pi/l4/kernel/Adam/Initializer/zeros/ConstConst*
_class
loc:@pi/l4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ţ
*atrain/pi/l4/kernel/Adam/Initializer/zerosFill:atrain/pi/l4/kernel/Adam/Initializer/zeros/shape_as_tensor0atrain/pi/l4/kernel/Adam/Initializer/zeros/Const*
T0*
_class
loc:@pi/l4/kernel*

index_type0* 
_output_shapes
:

ą
atrain/pi/l4/kernel/Adam
VariableV2*
shared_name *
_class
loc:@pi/l4/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ä
atrain/pi/l4/kernel/Adam/AssignAssignatrain/pi/l4/kernel/Adam*atrain/pi/l4/kernel/Adam/Initializer/zeros*
_class
loc:@pi/l4/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

atrain/pi/l4/kernel/Adam/readIdentityatrain/pi/l4/kernel/Adam* 
_output_shapes
:
*
T0*
_class
loc:@pi/l4/kernel
Ž
<atrain/pi/l4/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
_class
loc:@pi/l4/kernel*
valueB"      *
dtype0

2atrain/pi/l4/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@pi/l4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,atrain/pi/l4/kernel/Adam_1/Initializer/zerosFill<atrain/pi/l4/kernel/Adam_1/Initializer/zeros/shape_as_tensor2atrain/pi/l4/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*
_class
loc:@pi/l4/kernel*

index_type0
ł
atrain/pi/l4/kernel/Adam_1
VariableV2*
shared_name *
_class
loc:@pi/l4/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ę
!atrain/pi/l4/kernel/Adam_1/AssignAssignatrain/pi/l4/kernel/Adam_1,atrain/pi/l4/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@pi/l4/kernel

atrain/pi/l4/kernel/Adam_1/readIdentityatrain/pi/l4/kernel/Adam_1*
_class
loc:@pi/l4/kernel* 
_output_shapes
:
*
T0

(atrain/pi/l4/bias/Adam/Initializer/zerosConst*
_class
loc:@pi/l4/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ł
atrain/pi/l4/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l4/bias
×
atrain/pi/l4/bias/Adam/AssignAssignatrain/pi/l4/bias/Adam(atrain/pi/l4/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@pi/l4/bias*
validate_shape(

atrain/pi/l4/bias/Adam/readIdentityatrain/pi/l4/bias/Adam*
T0*
_class
loc:@pi/l4/bias*
_output_shapes	
:

*atrain/pi/l4/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/l4/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ľ
atrain/pi/l4/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@pi/l4/bias*
	container *
shape:
Ý
atrain/pi/l4/bias/Adam_1/AssignAssignatrain/pi/l4/bias/Adam_1*atrain/pi/l4/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/l4/bias*
validate_shape(*
_output_shapes	
:

atrain/pi/l4/bias/Adam_1/readIdentityatrain/pi/l4/bias/Adam_1*
T0*
_class
loc:@pi/l4/bias*
_output_shapes	
:
 
)atrain/pi/a/kernel/Adam/Initializer/zerosConst*
_class
loc:@pi/a/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
­
atrain/pi/a/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *
_class
loc:@pi/a/kernel*
	container *
shape:	
ß
atrain/pi/a/kernel/Adam/AssignAssignatrain/pi/a/kernel/Adam)atrain/pi/a/kernel/Adam/Initializer/zeros*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@pi/a/kernel*
validate_shape(

atrain/pi/a/kernel/Adam/readIdentityatrain/pi/a/kernel/Adam*
T0*
_class
loc:@pi/a/kernel*
_output_shapes
:	
˘
+atrain/pi/a/kernel/Adam_1/Initializer/zerosConst*
_class
loc:@pi/a/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Ż
atrain/pi/a/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *
_class
loc:@pi/a/kernel*
	container *
shape:	
ĺ
 atrain/pi/a/kernel/Adam_1/AssignAssignatrain/pi/a/kernel/Adam_1+atrain/pi/a/kernel/Adam_1/Initializer/zeros*
T0*
_class
loc:@pi/a/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(

atrain/pi/a/kernel/Adam_1/readIdentityatrain/pi/a/kernel/Adam_1*
_output_shapes
:	*
T0*
_class
loc:@pi/a/kernel

'atrain/pi/a/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
_class
loc:@pi/a/bias*
valueB*    *
dtype0

atrain/pi/a/bias/Adam
VariableV2*
_class
loc:@pi/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ň
atrain/pi/a/bias/Adam/AssignAssignatrain/pi/a/bias/Adam'atrain/pi/a/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
:

atrain/pi/a/bias/Adam/readIdentityatrain/pi/a/bias/Adam*
_output_shapes
:*
T0*
_class
loc:@pi/a/bias

)atrain/pi/a/bias/Adam_1/Initializer/zerosConst*
_class
loc:@pi/a/bias*
valueB*    *
dtype0*
_output_shapes
:
Ą
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
Ř
atrain/pi/a/bias/Adam_1/AssignAssignatrain/pi/a/bias/Adam_1)atrain/pi/a/bias/Adam_1/Initializer/zeros*
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(

atrain/pi/a/bias/Adam_1/readIdentityatrain/pi/a/bias/Adam_1*
_class
loc:@pi/a/bias*
_output_shapes
:*
T0
¨
-atrain/pi/dense/kernel/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
ľ
atrain/pi/dense/kernel/Adam
VariableV2*"
_class
loc:@pi/dense/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 
ď
"atrain/pi/dense/kernel/Adam/AssignAssignatrain/pi/dense/kernel/Adam-atrain/pi/dense/kernel/Adam/Initializer/zeros*
_output_shapes
:	*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(

 atrain/pi/dense/kernel/Adam/readIdentityatrain/pi/dense/kernel/Adam*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	
Ş
/atrain/pi/dense/kernel/Adam_1/Initializer/zerosConst*"
_class
loc:@pi/dense/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
ˇ
atrain/pi/dense/kernel/Adam_1
VariableV2*
_output_shapes
:	*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	*
dtype0
ő
$atrain/pi/dense/kernel/Adam_1/AssignAssignatrain/pi/dense/kernel/Adam_1/atrain/pi/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	

"atrain/pi/dense/kernel/Adam_1/readIdentityatrain/pi/dense/kernel/Adam_1*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	

+atrain/pi/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
§
atrain/pi/dense/bias/Adam
VariableV2*
shared_name * 
_class
loc:@pi/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
â
 atrain/pi/dense/bias/Adam/AssignAssignatrain/pi/dense/bias/Adam+atrain/pi/dense/bias/Adam/Initializer/zeros*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(

atrain/pi/dense/bias/Adam/readIdentityatrain/pi/dense/bias/Adam* 
_class
loc:@pi/dense/bias*
_output_shapes
:*
T0

-atrain/pi/dense/bias/Adam_1/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Š
atrain/pi/dense/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@pi/dense/bias
č
"atrain/pi/dense/bias/Adam_1/AssignAssignatrain/pi/dense/bias/Adam_1-atrain/pi/dense/bias/Adam_1/Initializer/zeros* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

 atrain/pi/dense/bias/Adam_1/readIdentityatrain/pi/dense/bias/Adam_1*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
:
^
atrain/Adam/learning_rateConst*
valueB
 *ˇŃ8*
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
 *wž?*
dtype0*
_output_shapes
: 
X
atrain/Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
ł
)atrain/Adam/update_pi/l1/kernel/ApplyAdam	ApplyAdampi/l1/kernelatrain/pi/l1/kernel/Adamatrain/pi/l1/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l1/MatMul_grad/tuple/control_dependency_1*
_class
loc:@pi/l1/kernel*
use_nesterov( *
_output_shapes
:	*
use_locking( *
T0
Ś
'atrain/Adam/update_pi/l1/bias/ApplyAdam	ApplyAdam
pi/l1/biasatrain/pi/l1/bias/Adamatrain/pi/l1/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*
_class
loc:@pi/l1/bias
´
)atrain/Adam/update_pi/l2/kernel/ApplyAdam	ApplyAdampi/l2/kernelatrain/pi/l2/kernel/Adamatrain/pi/l2/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l2/MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@pi/l2/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( 
Ś
'atrain/Adam/update_pi/l2/bias/ApplyAdam	ApplyAdam
pi/l2/biasatrain/pi/l2/bias/Adamatrain/pi/l2/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@pi/l2/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
´
)atrain/Adam/update_pi/l3/kernel/ApplyAdam	ApplyAdampi/l3/kernelatrain/pi/l3/kernel/Adamatrain/pi/l3/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l3/kernel*
use_nesterov( * 
_output_shapes
:

Ś
'atrain/Adam/update_pi/l3/bias/ApplyAdam	ApplyAdam
pi/l3/biasatrain/pi/l3/bias/Adamatrain/pi/l3/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l3/bias*
use_nesterov( *
_output_shapes	
:
´
)atrain/Adam/update_pi/l4/kernel/ApplyAdam	ApplyAdampi/l4/kernelatrain/pi/l4/kernel/Adamatrain/pi/l4/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/l4/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/l4/kernel*
use_nesterov( * 
_output_shapes
:

Ś
'atrain/Adam/update_pi/l4/bias/ApplyAdam	ApplyAdam
pi/l4/biasatrain/pi/l4/bias/Adamatrain/pi/l4/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon>atrain/gradients/pi/l4/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@pi/l4/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
­
(atrain/Adam/update_pi/a/kernel/ApplyAdam	ApplyAdampi/a/kernelatrain/pi/a/kernel/Adamatrain/pi/a/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon<atrain/gradients/pi/a/MatMul_grad/tuple/control_dependency_1*
_class
loc:@pi/a/kernel*
use_nesterov( *
_output_shapes
:	*
use_locking( *
T0

&atrain/Adam/update_pi/a/bias/ApplyAdam	ApplyAdam	pi/a/biasatrain/pi/a/bias/Adamatrain/pi/a/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon=atrain/gradients/pi/a/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pi/a/bias*
use_nesterov( *
_output_shapes
:
Ĺ
,atrain/Adam/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelatrain/pi/dense/kernel/Adamatrain/pi/dense/kernel/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilon@atrain/gradients/pi/dense/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	*
use_locking( *
T0*"
_class
loc:@pi/dense/kernel*
use_nesterov( 
ˇ
*atrain/Adam/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biasatrain/pi/dense/bias/Adamatrain/pi/dense/bias/Adam_1atrain/beta1_power/readatrain/beta2_power/readatrain/Adam/learning_rateatrain/Adam/beta1atrain/Adam/beta2atrain/Adam/epsilonAatrain/gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@pi/dense/bias*
use_nesterov( *
_output_shapes
:

atrain/Adam/mulMulatrain/beta1_power/readatrain/Adam/beta1'^atrain/Adam/update_pi/a/bias/ApplyAdam)^atrain/Adam/update_pi/a/kernel/ApplyAdam+^atrain/Adam/update_pi/dense/bias/ApplyAdam-^atrain/Adam/update_pi/dense/kernel/ApplyAdam(^atrain/Adam/update_pi/l1/bias/ApplyAdam*^atrain/Adam/update_pi/l1/kernel/ApplyAdam(^atrain/Adam/update_pi/l2/bias/ApplyAdam*^atrain/Adam/update_pi/l2/kernel/ApplyAdam(^atrain/Adam/update_pi/l3/bias/ApplyAdam*^atrain/Adam/update_pi/l3/kernel/ApplyAdam(^atrain/Adam/update_pi/l4/bias/ApplyAdam*^atrain/Adam/update_pi/l4/kernel/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@pi/a/bias
Š
atrain/Adam/AssignAssignatrain/beta1_poweratrain/Adam/mul*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@pi/a/bias*
validate_shape(

atrain/Adam/mul_1Mulatrain/beta2_power/readatrain/Adam/beta2'^atrain/Adam/update_pi/a/bias/ApplyAdam)^atrain/Adam/update_pi/a/kernel/ApplyAdam+^atrain/Adam/update_pi/dense/bias/ApplyAdam-^atrain/Adam/update_pi/dense/kernel/ApplyAdam(^atrain/Adam/update_pi/l1/bias/ApplyAdam*^atrain/Adam/update_pi/l1/kernel/ApplyAdam(^atrain/Adam/update_pi/l2/bias/ApplyAdam*^atrain/Adam/update_pi/l2/kernel/ApplyAdam(^atrain/Adam/update_pi/l3/bias/ApplyAdam*^atrain/Adam/update_pi/l3/kernel/ApplyAdam(^atrain/Adam/update_pi/l4/bias/ApplyAdam*^atrain/Adam/update_pi/l4/kernel/ApplyAdam*
T0*
_class
loc:@pi/a/bias*
_output_shapes
: 
­
atrain/Adam/Assign_1Assignatrain/beta2_poweratrain/Adam/mul_1*
use_locking( *
T0*
_class
loc:@pi/a/bias*
validate_shape(*
_output_shapes
: 
Ç
atrain/AdamNoOp^atrain/Adam/Assign^atrain/Adam/Assign_1'^atrain/Adam/update_pi/a/bias/ApplyAdam)^atrain/Adam/update_pi/a/kernel/ApplyAdam+^atrain/Adam/update_pi/dense/bias/ApplyAdam-^atrain/Adam/update_pi/dense/kernel/ApplyAdam(^atrain/Adam/update_pi/l1/bias/ApplyAdam*^atrain/Adam/update_pi/l1/kernel/ApplyAdam(^atrain/Adam/update_pi/l2/bias/ApplyAdam*^atrain/Adam/update_pi/l2/kernel/ApplyAdam(^atrain/Adam/update_pi/l3/bias/ApplyAdam*^atrain/Adam/update_pi/l3/kernel/ApplyAdam(^atrain/Adam/update_pi/l4/bias/ApplyAdam*^atrain/Adam/update_pi/l4/kernel/ApplyAdam""Ż
trainable_variables
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

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
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08"(
train_op

critic/Adam
atrain/Adam"ÓL
	variablesĹLÂL
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

critic/dense/kernel:0critic/dense/kernel/Assigncritic/dense/kernel/read:020critic/dense/kernel/Initializer/random_uniform:08
r
critic/dense/bias:0critic/dense/bias/Assigncritic/dense/bias/read:02%critic/dense/bias/Initializer/zeros:08
p
critic/beta1_power:0critic/beta1_power/Assigncritic/beta1_power/read:02"critic/beta1_power/initial_value:0
p
critic/beta2_power:0critic/beta2_power/Assigncritic/beta2_power/read:02"critic/beta2_power/initial_value:0

critic/critic/w1_s/Adam:0critic/critic/w1_s/Adam/Assigncritic/critic/w1_s/Adam/read:02+critic/critic/w1_s/Adam/Initializer/zeros:0

critic/critic/w1_s/Adam_1:0 critic/critic/w1_s/Adam_1/Assign critic/critic/w1_s/Adam_1/read:02-critic/critic/w1_s/Adam_1/Initializer/zeros:0

critic/critic/b1/Adam:0critic/critic/b1/Adam/Assigncritic/critic/b1/Adam/read:02)critic/critic/b1/Adam/Initializer/zeros:0

critic/critic/b1/Adam_1:0critic/critic/b1/Adam_1/Assigncritic/critic/b1/Adam_1/read:02+critic/critic/b1/Adam_1/Initializer/zeros:0

critic/critic/l2/kernel/Adam:0#critic/critic/l2/kernel/Adam/Assign#critic/critic/l2/kernel/Adam/read:020critic/critic/l2/kernel/Adam/Initializer/zeros:0
¤
 critic/critic/l2/kernel/Adam_1:0%critic/critic/l2/kernel/Adam_1/Assign%critic/critic/l2/kernel/Adam_1/read:022critic/critic/l2/kernel/Adam_1/Initializer/zeros:0

critic/critic/l2/bias/Adam:0!critic/critic/l2/bias/Adam/Assign!critic/critic/l2/bias/Adam/read:02.critic/critic/l2/bias/Adam/Initializer/zeros:0

critic/critic/l2/bias/Adam_1:0#critic/critic/l2/bias/Adam_1/Assign#critic/critic/l2/bias/Adam_1/read:020critic/critic/l2/bias/Adam_1/Initializer/zeros:0

critic/critic/l3/kernel/Adam:0#critic/critic/l3/kernel/Adam/Assign#critic/critic/l3/kernel/Adam/read:020critic/critic/l3/kernel/Adam/Initializer/zeros:0
¤
 critic/critic/l3/kernel/Adam_1:0%critic/critic/l3/kernel/Adam_1/Assign%critic/critic/l3/kernel/Adam_1/read:022critic/critic/l3/kernel/Adam_1/Initializer/zeros:0

critic/critic/l3/bias/Adam:0!critic/critic/l3/bias/Adam/Assign!critic/critic/l3/bias/Adam/read:02.critic/critic/l3/bias/Adam/Initializer/zeros:0

critic/critic/l3/bias/Adam_1:0#critic/critic/l3/bias/Adam_1/Assign#critic/critic/l3/bias/Adam_1/read:020critic/critic/l3/bias/Adam_1/Initializer/zeros:0
¨
!critic/critic/dense/kernel/Adam:0&critic/critic/dense/kernel/Adam/Assign&critic/critic/dense/kernel/Adam/read:023critic/critic/dense/kernel/Adam/Initializer/zeros:0
°
#critic/critic/dense/kernel/Adam_1:0(critic/critic/dense/kernel/Adam_1/Assign(critic/critic/dense/kernel/Adam_1/read:025critic/critic/dense/kernel/Adam_1/Initializer/zeros:0
 
critic/critic/dense/bias/Adam:0$critic/critic/dense/bias/Adam/Assign$critic/critic/dense/bias/Adam/read:021critic/critic/dense/bias/Adam/Initializer/zeros:0
¨
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

atrain/pi/l1/kernel/Adam:0atrain/pi/l1/kernel/Adam/Assignatrain/pi/l1/kernel/Adam/read:02,atrain/pi/l1/kernel/Adam/Initializer/zeros:0

atrain/pi/l1/kernel/Adam_1:0!atrain/pi/l1/kernel/Adam_1/Assign!atrain/pi/l1/kernel/Adam_1/read:02.atrain/pi/l1/kernel/Adam_1/Initializer/zeros:0

atrain/pi/l1/bias/Adam:0atrain/pi/l1/bias/Adam/Assignatrain/pi/l1/bias/Adam/read:02*atrain/pi/l1/bias/Adam/Initializer/zeros:0

atrain/pi/l1/bias/Adam_1:0atrain/pi/l1/bias/Adam_1/Assignatrain/pi/l1/bias/Adam_1/read:02,atrain/pi/l1/bias/Adam_1/Initializer/zeros:0

atrain/pi/l2/kernel/Adam:0atrain/pi/l2/kernel/Adam/Assignatrain/pi/l2/kernel/Adam/read:02,atrain/pi/l2/kernel/Adam/Initializer/zeros:0

atrain/pi/l2/kernel/Adam_1:0!atrain/pi/l2/kernel/Adam_1/Assign!atrain/pi/l2/kernel/Adam_1/read:02.atrain/pi/l2/kernel/Adam_1/Initializer/zeros:0

atrain/pi/l2/bias/Adam:0atrain/pi/l2/bias/Adam/Assignatrain/pi/l2/bias/Adam/read:02*atrain/pi/l2/bias/Adam/Initializer/zeros:0

atrain/pi/l2/bias/Adam_1:0atrain/pi/l2/bias/Adam_1/Assignatrain/pi/l2/bias/Adam_1/read:02,atrain/pi/l2/bias/Adam_1/Initializer/zeros:0

atrain/pi/l3/kernel/Adam:0atrain/pi/l3/kernel/Adam/Assignatrain/pi/l3/kernel/Adam/read:02,atrain/pi/l3/kernel/Adam/Initializer/zeros:0

atrain/pi/l3/kernel/Adam_1:0!atrain/pi/l3/kernel/Adam_1/Assign!atrain/pi/l3/kernel/Adam_1/read:02.atrain/pi/l3/kernel/Adam_1/Initializer/zeros:0

atrain/pi/l3/bias/Adam:0atrain/pi/l3/bias/Adam/Assignatrain/pi/l3/bias/Adam/read:02*atrain/pi/l3/bias/Adam/Initializer/zeros:0

atrain/pi/l3/bias/Adam_1:0atrain/pi/l3/bias/Adam_1/Assignatrain/pi/l3/bias/Adam_1/read:02,atrain/pi/l3/bias/Adam_1/Initializer/zeros:0

atrain/pi/l4/kernel/Adam:0atrain/pi/l4/kernel/Adam/Assignatrain/pi/l4/kernel/Adam/read:02,atrain/pi/l4/kernel/Adam/Initializer/zeros:0

atrain/pi/l4/kernel/Adam_1:0!atrain/pi/l4/kernel/Adam_1/Assign!atrain/pi/l4/kernel/Adam_1/read:02.atrain/pi/l4/kernel/Adam_1/Initializer/zeros:0

atrain/pi/l4/bias/Adam:0atrain/pi/l4/bias/Adam/Assignatrain/pi/l4/bias/Adam/read:02*atrain/pi/l4/bias/Adam/Initializer/zeros:0

atrain/pi/l4/bias/Adam_1:0atrain/pi/l4/bias/Adam_1/Assignatrain/pi/l4/bias/Adam_1/read:02,atrain/pi/l4/bias/Adam_1/Initializer/zeros:0

atrain/pi/a/kernel/Adam:0atrain/pi/a/kernel/Adam/Assignatrain/pi/a/kernel/Adam/read:02+atrain/pi/a/kernel/Adam/Initializer/zeros:0

atrain/pi/a/kernel/Adam_1:0 atrain/pi/a/kernel/Adam_1/Assign atrain/pi/a/kernel/Adam_1/read:02-atrain/pi/a/kernel/Adam_1/Initializer/zeros:0

atrain/pi/a/bias/Adam:0atrain/pi/a/bias/Adam/Assignatrain/pi/a/bias/Adam/read:02)atrain/pi/a/bias/Adam/Initializer/zeros:0

atrain/pi/a/bias/Adam_1:0atrain/pi/a/bias/Adam_1/Assignatrain/pi/a/bias/Adam_1/read:02+atrain/pi/a/bias/Adam_1/Initializer/zeros:0

atrain/pi/dense/kernel/Adam:0"atrain/pi/dense/kernel/Adam/Assign"atrain/pi/dense/kernel/Adam/read:02/atrain/pi/dense/kernel/Adam/Initializer/zeros:0
 
atrain/pi/dense/kernel/Adam_1:0$atrain/pi/dense/kernel/Adam_1/Assign$atrain/pi/dense/kernel/Adam_1/read:021atrain/pi/dense/kernel/Adam_1/Initializer/zeros:0

atrain/pi/dense/bias/Adam:0 atrain/pi/dense/bias/Adam/Assign atrain/pi/dense/bias/Adam/read:02-atrain/pi/dense/bias/Adam/Initializer/zeros:0

atrain/pi/dense/bias/Adam_1:0"atrain/pi/dense/bias/Adam_1/Assign"atrain/pi/dense/bias/Adam_1/read:02/atrain/pi/dense/bias/Adam_1/Initializer/zeros:0K¤