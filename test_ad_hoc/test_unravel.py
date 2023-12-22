import torch as t
from alan_simplified.unravel_index import unravel_index

import unittest

######## ! NOTE ! ########
# This test requires a version of pytorch that has torch.unravel_index (which, at the time of writing, means a nightly release).
#
# This test checks that our local copy of torch.unravel_index matches the version from a nightly release of pytorch.
# https://github.com/pytorch/pytorch/blob/5292a92e03f6f33ba8363abcae708c943baaf275/torch/functional.py#L1692
# 
# This is a temporary workaround for the fact that torch.unravel_index is not currently in the stable version of pytorch.
# CHECK BACK in a few weeks/months to see if this is still necessary and if any bugs were found/changes were made
# that should be mirrored here.
#
#
#############################

class TestUnravel(unittest.TestCase):
    def test_unravel_index(self):
        # Test case 1
        indices = t.tensor([6, 1, 3, 8])
        shape = (3, 4)
        torch_output = t.unravel_index(indices, shape)
        local_output = unravel_index(indices, shape)
        self.assertTrue(len(torch_output) == len(local_output))
        for i in range(len(torch_output)):
            self.assertTrue(t.all(t.eq(torch_output[i], local_output[i])))

        # Test case 2
        indices = t.tensor([1, 5, 7])
        shape = (2, 3)
        torch_output = t.unravel_index(indices, shape)
        local_output = unravel_index(indices, shape)
        self.assertTrue(len(torch_output) == len(local_output))
        for i in range(len(torch_output)):
            self.assertTrue(t.all(t.eq(torch_output[i], local_output[i])))

        # Test case 3
        indices = t.tensor([0, 2, 4, 6])
        shape = (2, 2, 2)
        torch_output = t.unravel_index(indices, shape)
        local_output = unravel_index(indices, shape)
        self.assertTrue(len(torch_output) == len(local_output))
        for i in range(len(torch_output)):
            self.assertTrue(t.all(t.eq(torch_output[i], local_output[i])))

        # Test case 4
        indices = t.tensor([0, 1, 2, 3, 4, 5])
        shape = (2, 3)
        torch_output = t.unravel_index(indices, shape)
        local_output = unravel_index(indices, shape)
        self.assertTrue(len(torch_output) == len(local_output))
        for i in range(len(torch_output)):
            self.assertTrue(t.all(t.eq(torch_output[i], local_output[i])))

if __name__ == '__main__':
    unittest.main()