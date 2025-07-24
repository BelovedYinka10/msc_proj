import qrcode

email = "Safiullah.Khan@mmu.ac.uk"
cc_list = ["M.Al-Khalidi@mmu.ac.uk", "24837087@stu.mmu.ac.uk"]
subject = "Inquiry"
body = "Hi, I would like to know more about your solution."

# Join CCs into comma-separated string
cc = ",".join(cc_list)

# Construct and encode mailto link
mailto_link = f"mailto:{email}?cc={cc}&subject={subject}&body={body}"
mailto_link = mailto_link.replace(" ", "%20")  # basic URL encoding for spaces

# Generate QR code
qr = qrcode.make(mailto_link)
qr.save("email_qr.png")
print("QR code with multiple CCs saved as email_qr.png")
