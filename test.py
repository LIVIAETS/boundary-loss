#!/usr/bin/env python3

import unittest

import torch
import numpy as np

import utils

class TestDice(unittest.TestCase):
    def test_equal(self):
        t = torch.zeros((1, 100, 100), dtype=torch.int64)
        t[0, 40:60, 40:60] = 1

        c = utils.class2one_hot(t, K=2)

        self.assertEqual(utils.dice_coef(c, c)[0, 0], 1)

    def test_empty(self):
        t = torch.zeros((1, 100, 100), dtype=torch.int64)
        t[0, 40:60, 40:60] = 1

        c = utils.class2one_hot(t, K=2)

        self.assertEqual(utils.dice_coef(c, c)[0, 0], 1)

    def test_caca(self):
        t = torch.zeros((1, 100, 100), dtype=torch.int64)
        t[0, 40:60, 40:60] = 1

        c = utils.class2one_hot(t, K=2)
        z = torch.zeros_like(c)
        z[0, 1, ...] = 1

        self.assertEqual(utils.dice_coef(c, z, smooth=0)[0, 0], 0)  # Annoying to deal with the almost equal thing


class TestHausdorff(unittest.TestCase):
    def test_closure(self):
        t = torch.zeros((1, 256, 256), dtype=torch.int64)
        t[0, 50:60, :] = 1

        t2 = utils.class2one_hot(t, K=2)
        self.assertEqual(tuple(t2.shape), (1, 2, 256, 256))

        self.assertTrue(torch.equal(utils.hausdorff(t2, t2), torch.zeros((1, 2))))

    def test_empty(self):
        t = torch.zeros((1, 256, 256), dtype=torch.int64)

        t2 = utils.class2one_hot(t, K=2)
        self.assertEqual(tuple(t2.shape), (1, 2, 256, 256))

        self.assertTrue(torch.equal(utils.hausdorff(t2, t2), torch.zeros((1, 2))))

    def test_caca(self):
        t = torch.zeros((1, 256, 256), dtype=torch.int64)
        t[0, 50:60, :] = 1

        t2 = utils.class2one_hot(t, K=2)
        self.assertEqual(tuple(t2.shape), (1, 2, 256, 256))

        z = torch.zeros_like(t)
        z2 = utils.class2one_hot(z, K=2)

        diag = (256**2 + 256**2) ** 0.5
        # print(f"{diag=}")
        # print(f"{utils.hausdorff(z2, t2)=}")

        self.assertTrue(torch.equal(utils.hausdorff(z2, t2),
                                    torch.tensor([[60, diag]], dtype=torch.float32)))

    def test_proper(self):
        t = torch.zeros((1, 256, 256), dtype=torch.int64)
        t[0, 50:60, :] = 1

        t2 = utils.class2one_hot(t, K=2)
        self.assertEqual(tuple(t2.shape), (1, 2, 256, 256))

        z = torch.zeros_like(t)
        z[0, 80:90, :] = 1
        z2 = utils.class2one_hot(z, K=2)

        self.assertTrue(torch.equal(utils.hausdorff(z2, t2),
                                    torch.tensor([[30, 30]], dtype=torch.float32)))


class TestDistMap(unittest.TestCase):
    def test_closure(self):
        a = np.zeros((1, 256, 256))
        a[:, 50:60, :] = 1

        o = utils.class2one_hot(torch.Tensor(a).type(torch.int64), K=2).numpy()
        res = utils.one_hot2dist(o[0])
        self.assertEqual(res.shape, (2, 256, 256))

        neg = (res <= 0) * res

        self.assertEqual(neg.sum(), (o * res).sum())

    def test_full_coverage(self):
        a = np.zeros((1, 256, 256))
        a[:, 50:60, :] = 1

        o = utils.class2one_hot(torch.Tensor(a).type(torch.int64), K=2).numpy()
        res = utils.one_hot2dist(o[0])
        self.assertEqual(res.shape, (2, 256, 256))

        self.assertEqual((res[1] <= 0).sum(), a.sum())
        self.assertEqual((res[1] > 0).sum(), (1 - a).sum())

    def test_empty(self):
        a = np.zeros((1, 256, 256))

        o = utils.class2one_hot(torch.Tensor(a).type(torch.int64), K=2).numpy()
        res = utils.one_hot2dist(o[0])
        self.assertEqual(res.shape, (2, 256, 256))

        self.assertEqual(res[1].sum(), 0)
        self.assertEqual((res[0] <= 0).sum(), a.size)

    def test_max_dist(self):
        """
        The max dist for a box should be at the midle of the object, +-1
        """
        a = np.zeros((1, 256, 256))
        a[:, 1:254, 1:254] = 1

        o = utils.class2one_hot(torch.Tensor(a).type(torch.int64), K=2).numpy()
        res = utils.one_hot2dist(o[0])
        self.assertEqual(res.shape, (2, 256, 256))

        self.assertEqual(res[0].max(), 127)
        self.assertEqual(np.unravel_index(res[0].argmax(), (256, 256)), (127, 127))

        self.assertEqual(res[1].min(), -126)
        self.assertEqual(np.unravel_index(res[1].argmin(), (256, 256)), (127, 127))

    def test_border(self):
        """
        Make sure the border inside the object is 0 in the distance map
        """

        for l in range(3, 5):
            a = np.zeros((1, 25, 25))
            a[:, 3:3 + l, 3:3 + l] = 1

            o = utils.class2one_hot(torch.Tensor(a).type(torch.int64), K=2).numpy()
            res = utils.one_hot2dist(o[0])
            self.assertEqual(res.shape, (2, 25, 25))

            border = (res[1] == 0)

            self.assertEqual(border.sum(), 4 * (l - 1))


if __name__ == "__main__":
    unittest.main()
