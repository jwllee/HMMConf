package nl.tue.alignment.test;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.OutputStream;

import org.deckfour.xes.classification.XEventClassifier;
import org.deckfour.xes.extension.std.XConceptExtension;
import org.deckfour.xes.factory.XFactoryNaiveImpl;
import org.deckfour.xes.in.XesXmlGZIPParser;
import org.deckfour.xes.info.impl.XLogInfoImpl;
import org.deckfour.xes.model.XLog;
import org.deckfour.xes.model.XTrace;
import org.deckfour.xes.out.XesXmlSerializer;

public class MergeLogs {

	public static void main(String[] args) throws Exception {

		final String[] models = new String[] { "d53", "d62", "d63", "d64" };

		final String[] suffixes = new String[] { "rad1.xes", "rad1_10_noise.xes", "rad1_20_noise.xes",
				"rad1_30_noise.xes" };
		File folder = new File("C:\\temp\\partially_ordered_logs\\");

		for (final String model : models) {
			for (final String suffix : suffixes) {

				File[] files = folder.listFiles(new FilenameFilter() {

					public boolean accept(File dir, String name) {
						return name.endsWith(suffix + ".gz") && name.startsWith(model);
					}
				});

				XLog merged = new XFactoryNaiveImpl().createLog();
				XConceptExtension.instance().assignName(merged, model + "_" + suffix);

				XesXmlGZIPParser parser = new XesXmlGZIPParser();
				XEventClassifier eventClassifier = XLogInfoImpl.STANDARD_CLASSIFIER;
				merged.getClassifiers().add(eventClassifier);

				for (File f : files) {

					XLog log = parser.parse(f).get(0);
					for (XTrace trace : log) {
						merged.add((XTrace) trace.clone());
					}
				}

				XesXmlSerializer serializer = new XesXmlSerializer();

				OutputStream out = new BufferedOutputStream(
						new FileOutputStream(new File(folder + "\\" + model + "_" + suffix)));
				serializer.serialize(merged, out);
				out.close();

			}
		}

	}
}
