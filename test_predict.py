import unittest

from predict import CollectiveCosts, Node, build_graph, critical_path


class PredictionTest(unittest.TestCase):
    def test_overlap_and_final_join(self):
        parameters = [{'name':'a', 'ready_ms':1.,'bytes':8},
                      {'name':'b', 'ready_ms':3.,'bytes':8}]
        small = build_graph(parameters,8,lambda size:2.,1.)
        large = build_graph(parameters,16,lambda size:3.,1.)
        self.assertEqual(critical_path(small)[0],6.)
        self.assertEqual(critical_path(large)[0],7.)
        self.assertEqual(sum(n.duration_ms for n in small),8.)

    def test_rank_max_before_repeat_median(self):
        rows = [{'bytes':size,'repeat':repeat,'rank':rank,'world_size':2,'ms':value}
                for size, values in [(8,[(1,4),(5,2),(3,2)]),(16,[(2,8),(10,4),(6,4)])]
                for repeat, pair in enumerate(values)
                for rank,value in enumerate(pair)]
        cost = CollectiveCosts(rows,2)
        self.assertEqual(cost(8),4.)
        self.assertEqual(cost(12),6.)
        with self.assertRaises(ValueError):
            cost(32)

    def test_collective_waits_for_slowest_rank(self):
        parameters = [{'rank':0,'name':'a','ready_ms':1.,'bytes':8},
                      {'rank':1,'name':'a','ready_ms':5.,'bytes':8}]
        graph = build_graph(parameters,8,lambda size:2.,1.)
        duration, path, _ = critical_path(graph)
        self.assertEqual(duration,8.)
        self.assertEqual(path[0],'rank1:gradient:a')

    def test_concurrent_device_costs_are_not_pooled_with_idle_wall_times(self):
        rows = [{'bytes':size,'repeat':0,'rank':rank,'world_size':2,
                 'condition':condition,'ms':100.,'device_ms':value*size/8}
                for size in (8,16) for rank,value in enumerate((2.,3.))
                for condition in ('none','cube')]
        for row in rows:
            if row['condition']=='none':
                row['device_ms'] /= 2
        busy = CollectiveCosts(rows,2,'device_ms','cube')
        idle = CollectiveCosts(rows,2,'device_ms','none')
        self.assertEqual(busy(12),4.5)
        self.assertEqual(idle(12),2.25)


if __name__ == '__main__':
    unittest.main()
