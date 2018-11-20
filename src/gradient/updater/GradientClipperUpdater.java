package gradient.updater;

import javax.annotation.Nonnull;

public class GradientClipperUpdater implements IGradientBasedUpdater{

	private final @Nonnull IGradientBasedUpdater updater;
	private final double maxValue;
	private final double minValue;

	public GradientClipperUpdater(IGradientBasedUpdater updater, double maxValue) {
		this.updater = updater;
		this.maxValue = maxValue;
		this.minValue = -1*maxValue;
	}
	
	@Override
	public void update(double[] values, double[] gradients, double fintess, double value) {
		
		
		for(int i=0; i<gradients.length; i++) {
			if(gradients[i]>0) {
				gradients[i] =  Math.min(gradients[i], maxValue);
			}
			else {
				gradients[i] =  Math.max(gradients[i], minValue);
			}
		}
		
		updater.update(values, gradients, fintess, value);;
	}

}
