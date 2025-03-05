import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Line } from "react-chartjs-2";
import "chart.js/auto";

export default function HomePage() {
  return (
    <div className="bg-gray-900 text-white min-h-screen p-6">
      <nav className="bg-gray-800 p-4 text-center text-xl font-bold">
        Finance Agent - AI-Powered Stock Insights
      </nav>
      <div className="max-w-4xl mx-auto mt-6 text-center">
        <h1 className="text-3xl font-bold mb-4">Welcome to Finance Agent</h1>
        <p className="text-lg text-gray-300 mb-6">
          Your AI-powered financial assistant for stock market predictions, real-time news monitoring, 
          sentiment analysis, and personalized investment insights.
        </p>
        <Button className="bg-blue-500 text-lg px-6 py-3">Get Started</Button>
      </div>
    </div>
  );
}
