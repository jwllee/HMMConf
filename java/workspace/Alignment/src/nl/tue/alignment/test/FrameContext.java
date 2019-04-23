package nl.tue.alignment.test;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.util.Collection;
import java.util.concurrent.Executor;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.text.DefaultCaret;

import org.processmining.framework.connections.Connection;
import org.processmining.framework.connections.ConnectionCannotBeObtained;
import org.processmining.framework.connections.ConnectionManager;
import org.processmining.framework.plugin.PluginContext;
import org.processmining.framework.plugin.PluginContextID;
import org.processmining.framework.plugin.PluginDescriptor;
import org.processmining.framework.plugin.PluginExecutionResult;
import org.processmining.framework.plugin.PluginManager;
import org.processmining.framework.plugin.PluginParameterBinding;
import org.processmining.framework.plugin.ProMFuture;
import org.processmining.framework.plugin.Progress;
import org.processmining.framework.plugin.RecursiveCallException;
import org.processmining.framework.plugin.events.Logger.MessageLevel;
import org.processmining.framework.plugin.events.PluginLifeCycleEventListener.List;
import org.processmining.framework.plugin.events.ProgressEventListener.ListenerList;
import org.processmining.framework.plugin.impl.FieldSetException;
import org.processmining.framework.providedobjects.ProvidedObjectManager;
import org.processmining.framework.util.Pair;

public class FrameContext extends JFrame implements PluginContext, Progress {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6648553932079021909L;
	private JLabel label;
	private JTextArea text;
	private int maximum;
	private int minimum;
	private int value;
	private String caption;

	public FrameContext() {
		label = new JLabel();
		label.setText("Progress indicator stub");
		this.getContentPane().add(label, BorderLayout.NORTH);

		text = new JTextArea();
		DefaultCaret caret = (DefaultCaret) text.getCaret();
		caret.setUpdatePolicy(DefaultCaret.ALWAYS_UPDATE);
		JScrollPane scroll = new JScrollPane(text);
		scroll.setViewportView(text);
		this.getContentPane().add(scroll, BorderLayout.CENTER);
		text.setLineWrap(false);

		text.setMinimumSize(new Dimension(300, 400));

		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		pack();
		validate();
	}

	public PluginManager getPluginManager() {
		// TODO Auto-generated method stub
		return null;
	}

	public ProvidedObjectManager getProvidedObjectManager() {
		// TODO Auto-generated method stub
		return null;
	}

	public ConnectionManager getConnectionManager() {
		// TODO Auto-generated method stub
		return null;
	}

	public PluginContextID createNewPluginContextID() {
		// TODO Auto-generated method stub
		return null;
	}

	public void invokePlugin(PluginDescriptor plugin, int index, Object... objects) {
		// TODO Auto-generated method stub

	}

	public void invokeBinding(PluginParameterBinding binding, Object... objects) {
		// TODO Auto-generated method stub

	}

	public Class<? extends PluginContext> getPluginContextType() {
		// TODO Auto-generated method stub
		return null;
	}

	public <T, C extends Connection> Collection<T> tryToFindOrConstructAllObjects(Class<T> type,
			Class<C> connectionType, String role, Object... input) throws ConnectionCannotBeObtained {
		// TODO Auto-generated method stub
		return null;
	}

	public <T, C extends Connection> T tryToFindOrConstructFirstObject(Class<T> type, Class<C> connectionType,
			String role, Object... input) throws ConnectionCannotBeObtained {
		// TODO Auto-generated method stub
		return null;
	}

	public <T, C extends Connection> T tryToFindOrConstructFirstNamedObject(Class<T> type, String name,
			Class<C> connectionType, String role, Object... input) throws ConnectionCannotBeObtained {
		// TODO Auto-generated method stub
		return null;
	}

	public PluginContext createChildContext(String label) {
		// TODO Auto-generated method stub
		return null;
	}

	public Progress getProgress() {
		return this;
	}

	public ListenerList getProgressEventListeners() {
		// TODO Auto-generated method stub
		return null;
	}

	public List getPluginLifeCycleEventListeners() {
		// TODO Auto-generated method stub
		return null;
	}

	public PluginContextID getID() {
		// TODO Auto-generated method stub
		return null;
	}

	public String getLabel() {
		// TODO Auto-generated method stub
		return null;
	}

	public Pair<PluginDescriptor, Integer> getPluginDescriptor() {
		// TODO Auto-generated method stub
		return null;
	}

	public PluginContext getParentContext() {
		// TODO Auto-generated method stub
		return null;
	}

	public java.util.List<PluginContext> getChildContexts() {
		// TODO Auto-generated method stub
		return null;
	}

	public PluginExecutionResult getResult() {
		// TODO Auto-generated method stub
		return null;
	}

	public ProMFuture<?> getFutureResult(int i) {
		// TODO Auto-generated method stub
		return null;
	}

	public Executor getExecutor() {
		// TODO Auto-generated method stub
		return null;
	}

	public boolean isDistantChildOf(PluginContext context) {
		// TODO Auto-generated method stub
		return false;
	}

	public void setFuture(PluginExecutionResult resultToBe) {
		// TODO Auto-generated method stub

	}

	public void setPluginDescriptor(PluginDescriptor descriptor, int methodIndex)
			throws FieldSetException, RecursiveCallException {
		// TODO Auto-generated method stub

	}

	public boolean hasPluginDescriptorInPath(PluginDescriptor descriptor, int methodIndex) {
		// TODO Auto-generated method stub
		return false;
	}

	public void log(String message, MessageLevel level) {
		text.append(message);
		text.append("\n");
	}

	public void log(String message) {
		text.append(message);
		text.append("\n");
	}

	public void log(Throwable exception) {
		text.append(exception.getMessage());
		text.append("\n");
	}

	public org.processmining.framework.plugin.events.Logger.ListenerList getLoggingListeners() {
		// TODO Auto-generated method stub
		return null;
	}

	public PluginContext getRootContext() {
		// TODO Auto-generated method stub
		return null;
	}

	public boolean deleteChild(PluginContext child) {
		// TODO Auto-generated method stub
		return false;
	}

	public <T extends Connection> T addConnection(T c) {
		// TODO Auto-generated method stub
		return null;
	}

	public void clear() {
		// TODO Auto-generated method stub

	}

	public void setMinimum(int minimum) {
		this.minimum = minimum;
		label.setText(caption + ": " + minimum + " / " + value + " / " + maximum);
	}

	public void setMaximum(int maximum) {
		this.maximum = maximum;
		label.setText(caption + ": " + minimum + " / " + value + " / " + maximum);
	}

	public void setValue(int value) {
		this.value = value;
		label.setText(caption + ": " + minimum + " / " + value + " / " + maximum);
	}

	public void setCaption(String caption) {
		this.caption = caption;

	}

	public String getCaption() {
		return caption;
	}

	public int getValue() {
		return value;
	}

	public void inc() {
		value++;
		label.setText(caption + ": " + minimum + " / " + value + " / " + maximum);
	}

	public void setIndeterminate(boolean makeIndeterminate) {
		// TODO Auto-generated method stub

	}

	public boolean isIndeterminate() {
		// TODO Auto-generated method stub
		return false;
	}

	public int getMinimum() {
		return minimum;
	}

	public int getMaximum() {
		return maximum;
	}

	public boolean isCancelled() {
		// TODO Auto-generated method stub
		return false;
	}

	public void cancel() {
		// TODO Auto-generated method stub

	}

}
