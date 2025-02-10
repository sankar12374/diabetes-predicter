import { useState } from "react";
import { FaUserMd, FaVial, FaHeartbeat, FaWeight, FaStethoscope, FaFlask, FaTachometerAlt, FaCalendarAlt } from "react-icons/fa";

function App() {
  const [formData, setFormData] = useState({
    pregnancies: "",
    glucose: "",
    bloodPressure: "",
    skinThickness: "",
    insulin: "",
    bmi: "",
    diabetesPedigree: "",
    age: "",
  });

  const [result, setResult] = useState("");

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const features = Object.values(formData).map(Number);
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      const data = await response.json();
      setResult(data.prediction);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#E3F2FD]">
      <div className="w-full max-w-lg bg-white p-10 rounded-2xl shadow-xl">
        <h1 className="text-3xl font-bold text-center text-[#0D47A1] mb-6 flex items-center justify-center">
          <FaUserMd className="mr-2 text-4xl text-[#0D47A1]" /> Diabetes Prediction
        </h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          {[
            { name: "pregnancies", label: "Pregnancies", icon: <FaUserMd /> },
            { name: "glucose", label: "Glucose Level", icon: <FaVial /> },
            { name: "bloodPressure", label: "Blood Pressure", icon: <FaHeartbeat /> },
            { name: "skinThickness", label: "Skin Thickness", icon: <FaWeight /> },
            { name: "insulin", label: "Insulin Level", icon: <FaFlask /> },
            { name: "bmi", label: "BMI", icon: <FaTachometerAlt /> },
            { name: "diabetesPedigree", label: "Diabetes Pedigree", icon: <FaStethoscope /> },
            { name: "age", label: "Age", icon: <FaCalendarAlt /> },
          ].map((field, index) => (
            <div key={index} className="flex items-center bg-gray-100 p-3 rounded-md shadow-md">
              <div className="text-[#0D47A1] text-xl mr-3">{field.icon}</div>
              <input
                type="number"
                name={field.name}
                value={formData[field.name]}
                onChange={handleChange}
                placeholder={field.label}
                className="flex-1 px-3 py-2 rounded-md bg-gray-100 text-black focus:ring-2 focus:ring-[#0D47A1] focus:outline-none transition duration-300"
                required
              />
            </div>
          ))}

          <button
            type="submit"
            className="w-full py-3 bg-[#0D47A1] text-lg text-white font-bold rounded-lg shadow-md hover:bg-[#0A377D] transition duration-300"
          >
            Predict
          </button>
        </form>

        {result && (
          <div className="mt-6 p-5 text-center text-xl font-semibold bg-white text-[#333] rounded-xl shadow-lg transition duration-300">
            <span className="text-[#00467F] font-bold">Prediction: </span>
            <span className={result === "Diabetic" ? "text-red-600" : "text-green-600"}>{result}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
