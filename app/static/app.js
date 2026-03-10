const form = document.getElementById("lead-form");
const statusText = document.getElementById("lead-status");

if (form && statusText) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());

    const response = await fetch("/lead-capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    statusText.textContent = response.ok
      ? "Lead captured successfully."
      : "Failed to capture lead.";
  });
}
