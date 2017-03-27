import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
# (<tf.Tensor 'Const:0' shape=() dtype=float32>, <tf.Tensor 'Const_1:0' shape=() dtype=float32>)
sess = tf.Session()
print(sess.run([node1, node2]))
# [3.0, 4.0]

node3 = tf.add(node1, node2)
print("node3: ", node3)
# ('node3: ', <tf.Tensor 'Add:0' shape=() dtype=float32>)
print("sess.run(node3): ",sess.run(node3))
# ('sess.run(node3): ', 7.0)
