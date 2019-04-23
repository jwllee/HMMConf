package org.processmining.alignment.plugin;

import org.processmining.framework.plugin.annotations.KeepInProMCache;
import org.processmining.plugins.petrinet.replayer.annotations.PNReplayAlgorithm;

import nl.tue.alignment.ReplayerParameters;
import nl.tue.alignment.algorithms.ReplayAlgorithm.Debug;

@KeepInProMCache
@PNReplayAlgorithm(isBasic = true)
public class IterativeAStarPlugin extends AbstractAlignmentPlugin {

	public IterativeAStarPlugin() {
		super();
	}

	public String getHTMLInfo() {
		return "<html>This is an algorithm to calculate cost-based fitness between a log and a Petri net. <br/><br/>"
				+ "Given a trace and a Petri net, this algorithm "
				+ "return a matching between the trace and an allowed firing sequence of the net with the"
				+ "least deviation cost using a iterative deepening A* technique. The firing sequence has to reach proper "
				+ "termination of the net, specified by 1 final marking. <br/><br/>"
				+ "The algorithm guarantees optimal results.</html>";
	}

	public String toString() {
		return "Splitting replayer assuming at most 127 tokens in each place.";
	}

	@Override
	protected ReplayerParameters constructReplayParameters(int numThreads, boolean usePartialOrder,
			int maximumNumberOfStates) {
		ReplayerParameters replayParameters = new ReplayerParameters.IncrementalAStar(false, numThreads, false,
				Debug.NONE, Integer.MAX_VALUE, maximumNumberOfStates, Integer.MAX_VALUE, usePartialOrder, false);
		return replayParameters;
	}

}
