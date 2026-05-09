async function predictLoan() {

    const result = document.getElementById("result");

    result.innerHTML = "Predicting...";
    result.className = "";

    const payload = {

        no_of_dependents:
            parseInt(document.getElementById("no_of_dependents").value),

        education:
            document.getElementById("education").value,

        self_employed:
            document.getElementById("self_employed").value,

        income_annum:
            parseInt(document.getElementById("income_annum").value),

        loan_amount:
            parseInt(document.getElementById("loan_amount").value),

        loan_term:
            parseInt(document.getElementById("loan_term").value),

        cibil_score:
            parseInt(document.getElementById("cibil_score").value),

        residential_assets_value:
            parseInt(document.getElementById("residential_assets_value").value),

        commercial_assets_value:
            parseInt(document.getElementById("commercial_assets_value").value),

        luxury_assets_value:
            parseInt(document.getElementById("luxury_assets_value").value),

        bank_asset_value:
            parseInt(document.getElementById("bank_asset_value").value),

        exam_qualified:
            document.getElementById("exam_qualified").value,

        admission_type:
            document.getElementById("admission_type").value,

        model_name:
            document.getElementById("model_name").value
    };

    try {

        // =====================================
        // Replace with your Render URL
        // =====================================

        const response = await fetch(
            "https://loan.onrender.com/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(payload)
            }
        );

        const data = await response.json();

        // =====================================
        // Result UI
        // =====================================

        if (data.prediction === "Approved") {

            result.className = "approved";

        } else {

            result.className = "rejected";
        }

        result.innerHTML = `
            <h2>${data.prediction}</h2>

            <p>
                Confidence:
                <strong>${data.confidence}%</strong>
            </p>

            <p>
                Model Used:
                <strong>${payload.model_name}</strong>
            </p>
        `;

    } catch (error) {

        result.className = "rejected";

        result.innerHTML = `
            Error connecting to backend.
            Please check deployment.
        `;

        console.error(error);
    }
}