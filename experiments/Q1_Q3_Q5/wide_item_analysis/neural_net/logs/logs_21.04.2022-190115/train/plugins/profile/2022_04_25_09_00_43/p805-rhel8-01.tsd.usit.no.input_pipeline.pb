	????w4@????w4@!????w4@	C??????C??????!C??????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????w4@O?)??Y??A??2SZ+4@Y???x軻?*	p=
ף?`@2U
Iterator::Model::ParallelMapV2%Z?xZ~??!??b???7@)%Z?xZ~??1??b???7@:Preprocessing2F
Iterator::Model?Q?=???!??'?BF@)??5?e??1,?Ə??4@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??O?s'??!4???Ƅ1@)??O?s'??14???Ƅ1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??
?H???!?(ҵj?4@)P ?Ȓ9??1|4???0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate{Cr??!**=>??:@)|DL?$z??1듸?lz"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipK=By??!t:??>?K@)??Z???1x?4??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?e??
z?!???@`?@)?e??
z?1???@`?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[Υ?????!f!????<@)>u?Rz?g?1??ϵ;'@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9C??????I??? ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	O?)??Y??O?)??Y??!O?)??Y??      ??!       "      ??!       *      ??!       2	??2SZ+4@??2SZ+4@!??2SZ+4@:      ??!       B      ??!       J	???x軻????x軻?!???x軻?R      ??!       Z	???x軻????x軻?!???x軻?b      ??!       JCPU_ONLYYC??????b q??? ?X@