package nl.tue.alignment.algorithms.syncproduct;

import java.util.ArrayList;

import org.deckfour.xes.model.XTrace;
import org.processmining.plugins.replayer.replayresult.SyncReplayResult;

import gnu.trove.map.TObjectIntMap;
import nl.tue.alignment.Replayer;
import nl.tue.alignment.Utils.Statistic;
import nl.tue.astar.Trace;

public interface SyncProductFactory<T> {

	public SyncProduct getSyncProduct(XTrace xTrace, ArrayList<? super T> transitionList,
			boolean partiallyOrderSameTimestamp);

	public SyncProduct getSyncProduct(XTrace xTrace, ArrayList<? super T> transitionList,
			boolean partiallyOrderSameTimestamp, boolean isPrefix);
	
	public SyncProduct getSyncProductForEmptyTrace(ArrayList<? super T> transitionList);

	public Trace getTrace(XTrace xTrace, boolean partiallyOrderSameTimestamp);

	public SyncReplayResult toSyncReplayResult(Replayer replayer, SyncProduct product,
			TObjectIntMap<Statistic> statistics, int[] alignment, XTrace trace, int traceIndex,
			ArrayList<? super T> transitionList);
}
