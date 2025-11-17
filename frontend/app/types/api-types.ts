export type feedbackApi = { feedback_prediction: string };
export type salesApi = { high_sales_prediction: number };
export type monthlySalesApi = { rf_sales_prediction: number };
export type ratingApi = { rf_rating_prediction: number };
export type successProbabilityApi = {
  success_probability_percentage: number;
  is_successful: boolean;
};
export type cityRecommendationApi = {
  top_3_recommendations: {
    city: string;
    probability_percent: number;
  }[];
};
export type monthRecommendationApi = {
  top_3_month_recommendations: {
    month: string;
    probability_percent: number;
  }[];
};
export type healthApi = {
  status: string;
  moels_loaded: string[];
};
