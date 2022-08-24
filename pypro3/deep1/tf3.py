# 연산자와 기본 함수 

import tensorflow as tf
import numpy as np

x = tf.constant(7)
y = 3

# 삼항연산
result1 = tf.cond(x > y, lambda:tf.add(x,y), lambda:tf.subtract(x,y))
print(result1.numpy())

# case
f1 = lambda:tf.constant(1)
f2 = lambda:tf.constant(2)
a = tf.constant(3)
b = tf.constant(4)
result2 = tf.case([(tf.less(a, b), f1)], default=f2)  # if a < b return 1 else return 2
print(result2.numpy())

print('관계 연산')
print(tf.equal(1, 2).numpy())
print(tf.not_equal(1,2))
print(tf.greater(1,2))
print(tf.less_equal(1,2))
print()
print('논리 연산')
print(tf.logical_and(True, False))
print(tf.logical_or(True, False))
print(tf.logical_not(True))
print()
print('차원 축소 함수')
ar = [[1.,2.],[3.,4.]]
print(tf.reduce_sum(ar).numpy())
print(tf.reduce_mean(ar).numpy())
print(tf.reduce_mean(ar, axis=0).numpy())  # 열기준
print(tf.reduce_mean(ar, axis=1).numpy())

# reshape:차원변경, squzze:차원축소, expand_dims:차원확대...










