package gradient.updater;

import javax.annotation.Nonnull;

public class GradientEnhancerUpdater implements IGradientBasedUpdater{

	private final @Nonnull IGradientBasedUpdater updater;
	private final double enhanceFactor;
	private final double limit;

	public GradientEnhancerUpdater(IGradientBasedUpdater updater, double enhanceFactor, double limit) {
		this.updater = updater;
		this.enhanceFactor = enhanceFactor;
		this.limit = limit;
	}
	
	@Override
	public void update(double[] values, double[] gradients, double fintess, double value) {
		for(int i=0; i<gradients.length; i++) {
			if(Math.abs(gradients[i])<limit) {
				gradients[i] *= enhanceFactor ;
			}
		}
		updater.update(values, gradients, fintess, value);;
	}

}
