package org.processmining.alignment.plugin;

import org.processmining.framework.plugin.annotations.KeepInProMCache;
import org.processmining.plugins.petrinet.replayer.annotations.PNReplayAlgorithm;

import nl.tue.alignment.ReplayerParameters;
import nl.tue.alignment.algorithms.ReplayAlgorithm.Debug;

@KeepInProMCache
@PNReplayAlgorithm(isBasic = true)
public class AStarPlugin extends AbstractAlignmentPlugin {

	public AStarPlugin() {
		super();
	}

	@Override
	protected ReplayerParameters constructReplayParameters(int numThreads, boolean usePartialOrder,
			int maximumNumberOfStates) {
		ReplayerParameters replayParameters = new ReplayerParameters.AStar(true, true, true, numThreads, false,
				Debug.NONE, Integer.MAX_VALUE, maximumNumberOfStates,Integer.MAX_VALUE, usePartialOrder);
		return replayParameters;
	}

	public String toString() {
		return "LP-based replayer assuming at most 127 tokens in each place.";
	}

	public String getHTMLInfo() {
		return "<html>This is an algorithm to calculate cost-based fitness between a log and a Petri net. <br/><br/>"
				+ "Given a trace and a Petri net, this algorithm "
				+ "return a matching between the trace and an allowed firing sequence of the net with the"
				+ "least deviation cost using A*. The firing sequence has to reach proper "
				+ "termination of the net, specified by 1 final marking. <br/><br/>"
				+ "The algorithm guarantees optimal results.</html>";
	}

}
